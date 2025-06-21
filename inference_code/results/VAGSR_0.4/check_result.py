# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Evaluation functions for BigBench Extra Hard."""

import json
import sys
import re
from typing import Dict, List, Any


def extract_answer(sample: str) -> str:
  """Extracts the final answer from the sample."""
  # 텍스트에서 (A), (B) 등의 패턴을 모두 찾아서 마지막 것을 반환
  matches = re.findall(r'\([A-Z]\)', sample)
  if matches:
    return matches[-1]  # 가장 마지막 매치 반환
  return sample


def fuzzy_match(prediction: str, reference: str) -> bool:
  """Fuzzy match function for BigBench Extra Hard."""
  # 대소문자 구분 없이 비교
  prediction = prediction.lower()
  reference = reference.lower()
  
  # 정확히 일치하는 경우
  if prediction == reference:
    return True

  # (A) vs A 형태 비교
  if len(prediction) == 3 and prediction[0] == "(" and prediction[-1] == ")":
    return prediction[1].lower() == reference.lower()
  if len(reference) == 3 and reference[0] == "(" and reference[-1] == ")":
    return reference[1].lower() == prediction.lower()

  return False


def preprocess_sample(sample: str) -> str:
  return extract_answer(sample.strip())


def preprocess_reference(reference: str) -> str:
  return reference.strip()


def evaluate_correctness(sample: str, reference: str) -> bool:
  prediction = preprocess_sample(sample)
  reference = preprocess_reference(reference)
  return fuzzy_match(prediction, reference)


def load_json_results(file_path: str) -> List[Dict[str, Any]]:
  """로드된 JSON 파일에서 결과를 가져옵니다."""
  with open(file_path, 'r') as f:
    results = json.load(f)
  return results


def evaluate_results(results: List[Dict[str, Any]]) -> Dict[str, Any]:
  """결과를 평가하고 통계를 반환합니다."""
  total = len(results)
  correct = 0
  
  details = []
  
  for item in results:
    sample = item["generated_text"]
    reference = item["ground_truth"]
    
    is_correct = evaluate_correctness(sample, reference)
    if is_correct:
      correct += 1
      
    details.append({
      "id": item["id"],
      "correct": is_correct,
      "prediction": preprocess_sample(sample),
      "reference": preprocess_reference(reference)
    })
  
  return {
    "total": total,
    "correct": correct,
    "accuracy": correct / total if total > 0 else 0,
    "details": details
  }


def main():
  """메인 함수로 JSON 파일을 로드하고 결과를 평가합니다."""
  if len(sys.argv) < 2:
    print("사용법: python check_result.py <결과_파일_경로>")
    return
  
  result_file = sys.argv[1]
  results = load_json_results(result_file)
  
  evaluation = evaluate_results(results)
  
  print(f"총 문제 수: {evaluation['total']}")
  print(f"정답 수: {evaluation['correct']}")
  print(f"정확도: {evaluation['accuracy']:.4f} ({evaluation['correct']}/{evaluation['total']})")
  
  # 자세한 결과 출력 (선택 사항)
  print("\n자세한 결과:")
  for detail in evaluation['details']:
    print(f"ID: {detail['id']}")
    print(f"정답 여부: {'O' if detail['correct'] else 'X'}")
    print(f"예측: {detail['prediction']}")
    print(f"참조: {detail['reference']}")
    print("-" * 50)


if __name__ == "__main__":
  # 예제 테스트 실행
  print("테스트 예제:")
  print(evaluate_correctness("따라서 정답은 (A) 입니다.", "(A)"))  # True
  print(evaluate_correctness("여러 선택지 중에서 (B), (A), (C) 순서로 검토했고 최종적으로 (C)가 정답입니다.", "(C)"))  # True
  print(evaluate_correctness("정답은 (D) 입니다.", "(B)"))  # False
  print(evaluate_correctness("(A)와 (B) 중에서 (A)가 더 적절합니다.", "(A)"))  # True
  
  # 메인 함수 실행
  main()