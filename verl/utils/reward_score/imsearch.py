# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import re
import string


# normalize_answer() & em_check() from Search-R1
# https://github.com/PeterGriffinJin/Search-R1
def normalize_answer(s):
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))

def em_check(prediction, golden_answers):
    '''
        prediction: string
        golden_answers: list or string, support multi candidate answers
    '''
    if isinstance(golden_answers, str):
        golden_answers = [golden_answers]
    normalized_prediction = normalize_answer(prediction)
    exactly_match = False
    for golden_answer in golden_answers:
        golden_answer = normalize_answer(golden_answer)
        if golden_answer == normalized_prediction:
            exactly_match = True
            break
    return exactly_match


def subem_check(prediction, golden_answers):
    if isinstance(golden_answers, str):
        golden_answers = [golden_answers]
    normalized_prediction = normalize_answer(prediction)
    score = 0
    for golden_answer in golden_answers:
        golden_answer = normalize_answer(golden_answer)
        if golden_answer in normalized_prediction:
            score = 1
            break
    return score

def extract_solution(prediction):
    """Extract the equation from the solution string."""

    answer_pattern = r'<answer>(.*?)</answer>'
    match = re.finditer(answer_pattern, prediction, re.DOTALL)
    matches = list(match)

    if not matches:
        return None
    else:
        return matches[-1].group(1).strip()

def format_reward(input_string, search_pattern):
    
    reason_count = len(re.findall(r"<reason>.*?</reason>", input_string))
    answer_count = len(re.findall(r"<answer>.*?</answer>", input_string))

    format_score = 0
    # call search tool
    if search_pattern in input_string:
        if reason_count == 2 and answer_count == 1:
            format_score = 1
    # direct answer
    else:
        if reason_count == 1 and answer_count == 1:
            format_score = 1
    
    return format_score
        
def compute_score(prediction: str, ground_truth: str, search_pattern="<search><img></search>"):
    
    # Exactly Match Scorer

    answer = extract_solution(prediction=prediction)
    score = 0
    # correctness check
    if answer is not None:
        if em_check(answer, ground_truth):
            score = 1
    # search check
    if search_pattern in prediction:
        score -= 0.1
    
    # Score: 0.9 * correctness + 0.1 * format
    # Correct: > 0.2 for robustness
    return 0.9*score + 0.1*(format_reward(prediction, search_pattern))