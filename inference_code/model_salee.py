import torch
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModel
from transformers.modeling_outputs import CausalLMOutput
import os
import logging
import faiss
import pickle
import numpy as np
from typing import Dict, List, Tuple, Union, Optional
from vllm import LLM

class Projector(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.linear1 = nn.Linear(4096, hidden_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.1)
        self.linear2 = nn.Linear(hidden_size, hidden_size)

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x

class VectorRagInferenceModel:
    def __init__(self, config) -> None:
        self.config = config
        self.device = torch.device(config.device)
        
        # RAG 정보 추적을 위한 속성
        self.rag_info = []
        
        # 프로젝터 (vector 변환)
        self.projector_ctx2llm = Projector(config.hidden_size)
        
        # 저장된 프로젝터 가중치 로드
        projector_path = os.path.join(config.parameter_path, config.projector_filename)
        if os.path.exists(projector_path):
            logging.info(f"Loading projector from {projector_path}")
            projector_weights = torch.load(projector_path, map_location=self.device)
            self.projector_ctx2llm.load_state_dict(projector_weights)
        else:
            logging.error(f"Projector weights not found at {projector_path}")
            raise FileNotFoundError(f"Projector weights not found at {projector_path}")
        
        self.projector_ctx2llm.to(self.device)
        self.projector_ctx2llm.eval()
        
        # vLLM을 사용하여 SFR-Embedding-Mistral 모델 로드 (단일 GPU만 사용)
        logging.info("Loading SFR-Embedding-Mistral model with vLLM (단일 GPU 모드)")
        self.sfr_model_name = "Salesforce/SFR-Embedding-Mistral"
        self.sfr_model = LLM(
            model=self.sfr_model_name,
            device=self.device.type + ":0",  # 명시적으로 0번 GPU 사용
            task="embed",
            trust_remote_code=True,
            seed=42
        )
        
        # 토크나이저에 특별 토큰 "[CTX]"를 추가
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name)
        special_tokens_dict = {"additional_special_tokens": ["[CTX]"]}
        num_added_tokens = self.tokenizer.add_special_tokens(special_tokens_dict)
        
        # base model을 불러오고, 추가된 토큰이 있다면 임베딩 크기를 재조정 (단일 GPU 모드)
        logging.info(f"Loading base model {config.model_name} in single GPU mode")
        self.model = AutoModelForCausalLM.from_pretrained(
            config.model_name,
            torch_dtype=torch.bfloat16,  # 메모리 사용량 감소
            low_cpu_mem_usage=True,      # CPU 메모리 사용 최소화
        )
        if num_added_tokens > 0:
            self.model.resize_token_embeddings(len(self.tokenizer))
        
        # 슬라이딩 윈도우 어텐션 문제 해결: eager 모드에서는 해당 기능 비활성화
        if hasattr(self.model.config, "use_sliding_window_attention") and self.model.config.use_sliding_window_attention:
            logging.info("Disabling sliding window attention in eager mode.")
            self.model.config.use_sliding_window_attention = False
            
        # 모델을 GPU로 이동
        self.model.to(self.device)
        self.model.eval()
        
        self.model.config.pad_token_id = self.tokenizer.pad_token_id
        self.tokenizer.padding_side = 'left'
        self.tokenizer.add_eos_token = True
        
        # FAISS 인덱스들 로드
        self.faiss_indexes = []
        self.faiss_metadata = []
        for i in range(1, 5):  # 12개의 FAISS 인덱스
            index_path = os.path.join(config.faiss_path, f"{i}_index.faiss")
            meta_path = os.path.join(config.faiss_path, f"{i}_metadata.pkl")
            
            if os.path.exists(index_path) and os.path.exists(meta_path):
                logging.info(f"Loading FAISS index {i} from {index_path}")
                index = faiss.read_index(index_path)
                with open(meta_path, "rb") as f:
                    metadata = pickle.load(f)
                self.faiss_indexes.append(index)
                self.faiss_metadata.append(metadata)
            else:
                logging.error(f"FAISS index or metadata not found: {index_path}, {meta_path}")
                raise FileNotFoundError(f"FAISS files not found: {index_path}, {meta_path}")
        
        logging.info(f"Loaded {len(self.faiss_indexes)} FAISS indexes")
    
    def get_detailed_instruct(self, query: str) -> str:
        """
        검색 쿼리를 위한 지시문 형식을 생성합니다.
        SFR 공식 문서에 따라 검색 쿼리에 지시문을 추가합니다.
        """
        task = "Given a web search query, retrieve relevant passages that answer the query"
        return f'Instruct: {task}\nQuery: {query}'
    
    def get_sfr_embeddings(self, text: str) -> np.ndarray:
        """
        vLLM의 SFR-Embedding-Mistral 모델을 사용하여 텍스트의 임베딩을 얻습니다.
        원본 Wikipedia 인덱스 생성에 사용된 방식과 동일한 방법으로 임베딩을 생성합니다.
        SFR 모델은 지시문이 추가된 형식의 입력을 예상합니다.
        """
        try:
            # 검색 쿼리 형식으로 변환 (지시문 추가)
            formatted_text = self.get_detailed_instruct(text)
            logging.info(f"Generating embedding for text: {formatted_text[:100]}...")
            
            # vLLM을 이용해 임베딩 생성
            result = self.sfr_model.encode(formatted_text, use_tqdm=False)
            embedding = result[0].outputs.embedding
            embedding = np.array(embedding, dtype=np.float32).reshape(1, -1)
            
            # L2 정규화 (정규화된 벡터의 내적은 코사인 유사도와 동일)
            faiss.normalize_L2(embedding)
            logging.info(f"Embedding shape: {embedding.shape}, norm: {np.linalg.norm(embedding)}")
            
            return embedding
        except Exception as e:
            logging.error(f"Error in get_sfr_embeddings: {e}")
            # 오류 발생 시 기본 임베딩 반환 (제로 벡터 대신 랜덤 벡터 사용)
            default_embedding = np.random.randn(1, 4096).astype(np.float32)
            faiss.normalize_L2(default_embedding)
            return default_embedding
    
    def get_embeddings(self, text: str) -> torch.Tensor:
        """
        LLM 모델용 임베딩을 얻습니다.
        모델 내부 처리에 사용됩니다.
        """
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True)
        
        # 마지막 레이어의 마지막 토큰 임베딩 사용
        last_hidden_state = outputs.hidden_states[-1]
        sentence_embedding = last_hidden_state.mean(dim=1)
        
        return sentence_embedding
    
    def search_faiss(self, query_embedding, k=10):
        """
        12개의 FAISS 인덱스에서 각각 k개의 문서를 검색하고 유사도 점수별로 정렬
        """
        all_results = []
        
        try:
            for idx, (index, meta) in enumerate(zip(self.faiss_indexes, self.faiss_metadata)):
                # FAISS 검색 수행
                D, I = index.search(query_embedding.reshape(1, -1), k)
                valid_results = len([i for i in I[0] if i != -1])
                logging.info(f"FAISS index {idx+1}: found {valid_results} valid results")
                
                # 결과 구성
                for i in range(len(I[0])):
                    if I[0][i] != -1:  # 유효한 인덱스인 경우
                        doc_idx = I[0][i]
                        similarity = float(D[0][i])
                        
                        # 메타데이터에서 안전하게 필드 추출
                        document = ''
                        if doc_idx in meta:
                            if 'text' in meta[doc_idx]:
                                document = meta[doc_idx]['text']
                            elif 'document' in meta[doc_idx]:
                                document = meta[doc_idx]['document']
                            elif 'chunk_text' in meta[doc_idx]:
                                document = meta[doc_idx]['chunk_text']
                        
                        # 임베딩 필드 가져오기 - 없으면 새로 생성
                        embedding = None
                        if doc_idx in meta and 'embedding' in meta[doc_idx]:
                            embedding = meta[doc_idx]['embedding']
                        
                        # 임베딩이 없으면 직접 생성
                        if embedding is None:
                            # 문서 텍스트가 있으면 임베딩 생성
                            if document:
                                try:
                                    # 문서 텍스트로 임베딩 생성
                                    logging.info(f"Creating embedding for document: {document[:100]}...")
                                    formatted_text = self.get_detailed_instruct(document)
                                    result = self.sfr_model.encode(formatted_text, use_tqdm=False)
                                    embedding = result[0].outputs.embedding
                                    embedding = np.array(embedding, dtype=np.float32)
                                    # L2 정규화
                                    faiss.normalize_L2(embedding.reshape(1, -1))
                                    embedding = embedding.flatten()
                                    logging.info(f"Created embedding with shape {embedding.shape}")
                                except Exception as e:
                                    logging.error(f"Error creating embedding: {e}")
                                    # 오류 시 랜덤 임베딩 생성
                                    embedding = np.random.randn(4096).astype(np.float32)
                                    faiss.normalize_L2(embedding.reshape(1, -1))
                                    embedding = embedding.flatten()
                            else:
                                # 문서가 없으면 랜덤 임베딩 생성
                                embedding = np.random.randn(4096).astype(np.float32)
                                faiss.normalize_L2(embedding.reshape(1, -1))
                                embedding = embedding.flatten()
                        
                        # 결과 추가 (embedding 필드가 없어도 결과 추가)
                        all_results.append({
                            'index_id': idx + 1,
                            'doc_id': doc_idx,
                            'similarity': similarity,
                            'document': document,
                            'embedding': embedding
                        })
                        logging.debug(f"Added result: index={idx+1}, doc_id={doc_idx}, similarity={similarity:.4f}, doc_length={len(document)}")
            
            # 유사도에 따라 정렬
            all_results.sort(key=lambda x: x['similarity'], reverse=True)
            logging.info(f"Total results after sorting: {len(all_results)}")
            
            # 결과가 없으면 로그 출력
            if not all_results:
                logging.warning(f"No search results found in any FAISS index!")
            
            # 상위 결과 내용 디버깅
            if all_results:
                top_result = all_results[0]
                logging.info(f"Top result - similarity: {top_result['similarity']:.4f}, document: {top_result['document'][:100]}...")
            
            return all_results
        except Exception as e:
            logging.error(f"Error in search_faiss: {e}", exc_info=True)
            return []
    
    def generate_with_vector_augmentation(self, input_text, max_length=512, temperature=0.7, confidence_threshold=0.3):
        """
        입력 텍스트로부터 생성하는 도중 문장 단위로 확신도를 측정하고, 
        확신도가 낮은 경우에만 FAISS를 이용해 관련 정보를 검색하여 augment합니다.
        </think> 토큰이 생성된 이후에는 Vector RAG를 중단합니다.
        
        Args:
            input_text: 입력 텍스트
            max_length: 최대 생성 길이
            temperature: 생성 시 temperature 값
            confidence_threshold: 확신도 임계값. 이 값보다 낮으면 FAISS 검색 수행
        """
        try:
            # 로깅 레벨을 INFO로 설정하여 더 자세한 디버깅 정보 표시 (기존과 동일)
            root_logger = logging.getLogger()
            original_level = root_logger.level
            root_logger.setLevel(logging.INFO)
            
            # RAG 정보 초기화
            self.rag_info = []
            
            logging.info(f"시작: Vector Augmented Generation (confidence_threshold={confidence_threshold})")
            logging.info(f"입력 텍스트: {input_text[:100]}...")
            
            # 초기 입력을 토큰화
            input_encoding = self.tokenizer(input_text, return_tensors="pt").to(self.device)
            input_ids = input_encoding.input_ids
            attention_mask = input_encoding.attention_mask
            
            logging.info(f"입력 토큰 수: {input_ids.size(1)}")
            
            # 생성을 위한 변수 초기화
            generated_tokens = input_ids.clone()
            gen_len = 0
            max_new_tokens = max_length - input_ids.size(1)
            
            # 문장 구분자 및 특수 토큰
            sentence_end_tokens = [self.tokenizer.encode('.')[1], self.tokenizer.encode('?')[1], self.tokenizer.encode('!')[1]]
            
            # </think> 토큰 ID 확인 (이 토큰 이후에 RAG 중단)
            think_end_token_ids = []
            try:
                think_end_token_ids = self.tokenizer.encode("</think>", add_special_tokens=False)
                if isinstance(think_end_token_ids, list) and len(think_end_token_ids) > 0:
                    logging.info(f"</think> token IDs: {think_end_token_ids}")
                else:
                    logging.warning("Failed to get </think> token ID")
            except Exception as e:
                logging.warning(f"Error getting </think> token ID: {e}")
            
            # [CTX] 토큰 ID 확인
            ctx_token_id = self.tokenizer.convert_tokens_to_ids("[CTX]")
            logging.info(f"[CTX] 토큰 ID: {ctx_token_id}")
            
            # 상태 변수
            recent_sentences = []
            
            # 현재 문장의 토큰별 확률 차이 저장
            current_sentence_prob_gaps = []
            
            # Vector RAG 활성화 여부
            vector_rag_enabled = True
            
            # [CTX] 토큰 추가 횟수 추적
            ctx_tokens_added = 0
            
            # 디버깅을 위한 RAG 정보 저장
            rag_info = []
            gen_len = 0
            
            # KV 캐시 최적화를 위한 과정
            with torch.no_grad():
                # 초기 KV 캐시 계산
                model_outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    use_cache=True
                )
                
                # 초기 KV 캐시 저장
                past_key_values = model_outputs.past_key_values
                
                # 토큰 단위 생성 (KV 캐시 활용)
                while gen_len < max_new_tokens:
                    # KV 캐시를 활용한 다음 토큰 예측
                    outputs = self.model(
                        input_ids=generated_tokens[:, -1:],  # 마지막 토큰만 입력
                        past_key_values=past_key_values,
                        use_cache=True
                    )
                    
                    # KV 캐시 업데이트
                    past_key_values = outputs.past_key_values
                    
                    next_token_logits = outputs.logits[:, -1, :]
                    
                    # 확률 계산
                    #probs = torch.nn.functional.softmax(next_token_logits / temperature, dim=-1)
                    probs = torch.nn.functional.softmax(next_token_logits, dim=-1)
                    # Top-2 확률 값과 인덱스 가져오기
                    top2_probs, top2_indices = torch.topk(probs, k=min(2, probs.size(-1)), dim=-1)
                    
                    # 가장 높은 확률과 두 번째 높은 확률의 차이 계산
                    if top2_probs.size(1) >= 2:
                        prob_gap = (top2_probs[0, 0] - top2_probs[0, 1]).item()
                        if gen_len % 5 == 0:  # 5 토큰마다 확률 차이 로깅
                            logging.info(f"토큰 {gen_len}: prob_gap = {prob_gap:.4f}, top1_prob = {top2_probs[0, 0]:.4f}, top1_token = '{self.tokenizer.decode([top2_indices[0, 0].item()])}'")
                    else:
                        prob_gap = 1.0  # 두 번째 확률이 없으면 최대 차이로 설정
                        
                    current_sentence_prob_gaps.append(prob_gap)
                    
                    # 샘플링
                    if temperature > 0:
                        next_token = torch.multinomial(probs, num_samples=1)
                    else:
                        next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
                    
                    # 다음 토큰 추가
                    generated_tokens = torch.cat([generated_tokens, next_token], dim=-1)
                    gen_len += 1
                    
                    # 100 토큰마다 현재 생성 상태 로깅
                    if gen_len % 100 == 0:
                        current_text = self.tokenizer.decode(generated_tokens[0], skip_special_tokens=False)
                        logging.info(f"생성 진행 ({gen_len} tokens): {current_text[-200:]}...")
                    
                    # </think> 토큰이 생성되었는지 확인
                    if think_end_token_ids and next_token.item() in think_end_token_ids:
                        vector_rag_enabled = False
                        logging.info("</think> 토큰 감지. Vector RAG 비활성화.")
                    
                    # 문장의 끝인지 확인 및 Vector RAG 수행 (RAG 활성화 상태에서만)
                    if next_token.item() in sentence_end_tokens and vector_rag_enabled:
                        # 현재까지 생성된 전체 텍스트 디코딩
                        current_text = self.tokenizer.decode(generated_tokens[0], skip_special_tokens=False)
                        
                        # 마지막 문장 추출 (새로 생성된 문장)
                        all_sentences = [s.strip() + "." for s in current_text.split('.') if s.strip()]
                        if all_sentences:
                            # 마지막 문장 추가
                            if len(all_sentences) > 0:
                                last_sentence = all_sentences[-1]
                                recent_sentences.append(last_sentence)
                                logging.info(f"새 문장 추가: '{last_sentence}'")
                        
                        # 현재 문장의 확신도 계산 (확률 차이의 평균)
                        if current_sentence_prob_gaps:
                            avg_prob_gap = sum(current_sentence_prob_gaps) / len(current_sentence_prob_gaps)
                            logging.info(f"문장 종료 확인: 평균 확신도 = {avg_prob_gap:.4f}, 임계값 = {confidence_threshold:.4f}")
                        else:
                            avg_prob_gap = 1.0  # 기본값 설정
                        
                        # 확신도가 임계값보다 낮고, 가용한 문장이 있으면 임베딩 검색 수행
                        if avg_prob_gap < confidence_threshold and len(recent_sentences) > 10:
                            logging.info(f"낮은 확신도 감지 ({avg_prob_gap:.4f} < {confidence_threshold:.4f}), FAISS 검색 수행 중...")
                            
                            # 최근 3개 문장만 유지 (또는 가용한 모든 문장)
                            recent_sentences = recent_sentences[-3:]
                            
                            # 문장들을 합쳐서 임베딩
                            combined_text = " ".join(recent_sentences)
                            logging.info(f"FAISS 검색용 쿼리 문장: '{combined_text}'")
                            
                            # SFR 모델로 문장 임베딩 구하기 (FAISS 검색용)
                            sentence_embedding = self.get_sfr_embeddings(combined_text)
                            
                            # FAISS 검색으로 관련 문서 찾기
                            search_results = self.search_faiss(sentence_embedding, k=10)
                            
                            # 상위 5개 결과만 사용
                            top_results = search_results[:5]
                            
                            if top_results:
                                try:
                                    # 가장 유사한 문서의 임베딩 가져오기
                                    top_embedding = torch.tensor(top_results[0]['embedding'], device=self.device)
                                    
                                    # 검색된 문서 내용 로깅
                                    logging.info(f"검색된 유사 문서 (similarity={top_results[0]['similarity']:.4f}): '{top_results[0]['document'][:200]}...'")
                                    
                                    # 프로젝터를 통해 임베딩 변환
                                    projected_embedding = self.projector_ctx2llm(top_embedding.unsqueeze(0))
                                    logging.info(f"프로젝터 통과 임베딩 노름: {torch.norm(projected_embedding).item():.4f}")
                                    
                                    # 문맥 증강을 위한 새 입력 구성
                                    # [CTX] 토큰을 추가하고, 이 위치에 벡터를 주입
                                    augmented_text = current_text + " [CTX]"
                                    
                                    # 새 입력 토큰화
                                    augmented_encoding = self.tokenizer(augmented_text, return_tensors="pt").to(self.device)
                                    aug_input_ids = augmented_encoding.input_ids
                                    attention_mask = torch.ones_like(aug_input_ids)
                                    
                                    # [CTX] 토큰 위치 찾기
                                    ctx_positions = (aug_input_ids == ctx_token_id).nonzero(as_tuple=True)
                                    
                                    # [CTX] 토큰이 없으면 계속 진행
                                    if ctx_positions[0].numel() == 0 or ctx_positions[1].numel() == 0:
                                        logging.warning("[CTX] 토큰을 입력에서 찾을 수 없습니다. 벡터 증강 건너뜀.")
                                        continue
                                    
                                    ctx_position = ctx_positions[1][0].item()
                                    logging.info(f"[CTX] 토큰 위치: {ctx_position}, 전체 토큰 수: {aug_input_ids.size(1)}")
                                    
                                    # 임베딩 공간에서 작업 수행
                                    input_embeds = self.model.get_input_embeddings()(aug_input_ids)
                                    
                                    # [CTX] 토큰 임베딩을 projected_embedding으로 대체하기 전에 원본 임베딩 정보 로깅
                                    ctx_orig_embed_norm = torch.norm(input_embeds[0, ctx_position]).item()
                                    logging.info(f"[CTX] 토큰 원본 임베딩 노름: {ctx_orig_embed_norm:.4f}")
                                    
                                    # [CTX] 토큰 위치에 projected_embedding으로 대체
                                    input_embeds[0, ctx_position] = projected_embedding[0, 0]
                                    
                                    # 대체 후 임베딩 정보 로깅
                                    ctx_new_embed_norm = torch.norm(input_embeds[0, ctx_position]).item()
                                    logging.info(f"[CTX] 토큰 대체 후 임베딩 노름: {ctx_new_embed_norm:.4f}")
                                    
                                    # 증강된 임베딩으로 생성 계속
                                    logging.info(f"벡터 증강 생성 시작 (총 {ctx_tokens_added+1}번째 [CTX])...")
                                    
                                    # 증강 정보 기록
                                    rag_info_entry = {
                                        "rag_index": ctx_tokens_added + 1,
                                        "position": gen_len,
                                        "confidence": avg_prob_gap,
                                        "context_before": current_text[-100:],
                                        "retrieved_doc": top_results[0]['document'][:100],
                                        "similarity_score": top_results[0]['similarity']
                                    }
                                    rag_info.append(rag_info_entry)
                                    self.rag_info.append(rag_info_entry)
                                    
                                    outputs = self.model.generate(
                                        inputs_embeds=input_embeds,
                                        attention_mask=attention_mask,
                                        max_length=generated_tokens.size(1) + 100,
                                        #temperature=temperature,
                                        do_sample=False,
                                        num_beams=1,
                                        pad_token_id=self.tokenizer.pad_token_id,
                                        eos_token_id=self.tokenizer.eos_token_id
                                    )
                                    
                                    # 생성 결과 확인
                                    augmented_result = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                                    logging.info(f"벡터 증강 후 생성 결과: '{augmented_result[-200:]}...'")
                                    
                                    # 생성된 텍스트에서 벡터 증강 이후 추가된 부분 식별 시도
                                    original_len = len(current_text)
                                    augmented_text_decoded = self.tokenizer.decode(outputs[0], skip_special_tokens=False)
                                    new_content = augmented_text_decoded[original_len+10:]  # [CTX] 토큰 길이 감안
                                    logging.info(f"[CTX] 이후 새로 생성된 텍스트: '{new_content[:100]}...'")
                                    
                                    # 새로 생성된 부분으로 기존 생성 결과 업데이트 - gen_len 값 누적
                                    # 기존 gen_len 값 저장
                                    previous_gen_len = gen_len

                                    generated_tokens = torch.cat([generated_tokens, outputs], dim=-1)
                                    
                                    # 증강된 입력(aug_input_ids)에 추가된 토큰 수 계산
                                    # 참고: model.generate의 outputs는 항상 전체 시퀀스(입력+생성)를 포함
                                    new_tokens_generated = max(0, generated_tokens.size(1) - aug_input_ids.size(1) + 1)
                                    
                                    # gen_len에 새로 생성된 토큰 수만 추가하여 누적
                                    gen_len = previous_gen_len + new_tokens_generated
                                    
                                    logging.info(f"토큰 계산: 기존 gen_len({previous_gen_len}) + 새로 생성된 토큰({new_tokens_generated}) = 현재 gen_len({gen_len})")
                                    
                                    # [CTX] 토큰 추가 횟수 증가
                                    ctx_tokens_added += 1
                                    
                                    # 로그에 기록
                                    logging.info(f"벡터 증강 성공! 총 [CTX] 토큰 추가 횟수: {ctx_tokens_added}")
                                    
                                    # KV 캐시 재계산 - 새로 생성된 tokens에 맞게 업데이트
                                    logging.info("Vector RAG 이후 KV 캐시 재계산 중...")
                                    with torch.no_grad():
                                        # 전체 시퀀스에 대해 KV 캐시 재계산
                                        model_outputs = self.model(
                                            input_ids=generated_tokens[:, :-1],
                                            use_cache=True
                                        )
                                        # KV 캐시 업데이트
                                        past_key_values = model_outputs.past_key_values
                                    logging.info("KV 캐시 재계산 완료")
                                    
                                    # 이전 문장 내역을 초기화하여 새로운 컨텍스트에서 시작
                                    recent_sentences = []
                                    
                                except Exception as e:
                                    logging.error(f"벡터 증강 중 오류 발생: {e}", exc_info=True)
                        
                        # 새 문장이 시작되므로 확률 차이 리스트 초기화
                        current_sentence_prob_gaps = []
                    
                    # EOS 토큰이 생성되면 중단
                    if next_token.item() == self.tokenizer.eos_token_id:
                        logging.info("EOS 토큰 생성, 생성 중단")
                        break
                    
                    # 최대 길이에 도달하면 종료
                    if gen_len >= max_new_tokens:
                        logging.info(f"최대 생성 길이 도달 ({max_new_tokens} 토큰), 생성 중단")
                        break
                
                # 최종 생성 결과 디코딩 (총 [CTX] 토큰 추가 횟수 로깅)
                generated_text = self.tokenizer.decode(generated_tokens[0], skip_special_tokens=False)
                
                # RAG 수행 정보 요약
                logging.info(f"생성 완료! 총 {ctx_tokens_added}번의 벡터 증강 수행")
                for i, info in enumerate(rag_info):
                    logging.info(f"RAG #{i+1}: 위치={info['position']}, 확신도={info['confidence']:.4f}")
                    logging.info(f"  컨텍스트: '{info['context_before']}'")
                    logging.info(f"  검색된 문서: '{info['retrieved_doc']}'")
                
                # 로깅 레벨 복원
                root_logger.setLevel(original_level)
                
                return generated_text
                
        except Exception as e:
            logging.error(f"generate_with_vector_augmentation 함수 오류: {e}", exc_info=True)
            # 오류 발생 시 입력 텍스트 반환
            return input_text + " [ERROR: Generation failed]" 
        