import csv
import glob
import sys
from dataclasses import dataclass

from dotenv import load_dotenv
from langchain.chains import LLMChain, SimpleSequentialChain
from langchain_core.prompts import PromptTemplate
from langchain_openai import OpenAI, ChatOpenAI

load_dotenv()

local = ChatOpenAI(
    openai_api_key="DUMMY",
    openai_api_base="http://localhost:1234/v1",
    model_name="DUMMY",
    streaming=False,
    temperature=0.3,
)

openai_gpt35 = ChatOpenAI(
    model_name="gpt-3.5-turbo",
    streaming=False,
    temperature=0.3,
)


def collect_files(glob_patterns: list[str]) -> list[str]:
    file_set = {file for pattern in glob_patterns for file in glob.glob(pattern)}
    return sorted(list(file_set))


def read_file(file_path: str) -> str | None:
    with open(file_path, "r") as file:
        lines = file.readlines()
        if not lines[0].startswith("<?php"):
            return None
        class_keyword_index = None
        for i, line in enumerate(lines):
            if line.startswith("class "):
                class_keyword_index = i
                break
        if class_keyword_index is None:
            return None
        return "".join(lines[class_keyword_index:])


@dataclass
class Result:
    file_path: str
    php_test_code: str
    result: str


def write_to_csv(results: list[Result], dest="output.csv") -> None:
    with open(dest, "w") as f:
        w = csv.writer(f)
        w.writerow(["file_path", "result"])
        for result in results:
            w.writerow([result.file_path, result.result])


def new_analyze_code_chain(llm: OpenAI = local):
    prompt = PromptTemplate.from_template(
        """Analyze following PHPUnit code and extract test methods that probably depend on existing database records, not using mocks or fixtures.
You must return only the method names as list `["testMethod1", "testMethod2", ...]` with no other information.
If no such method exists, you must return only `[]` otherwise.

INPUT:
{input}

OUTPUT:
"""
    )
    return LLMChain(llm=llm, prompt=prompt)


if __name__ == "__main__":
    glob_patterns = sys.argv[1:]
    files = collect_files(glob_patterns)
    php_test_codes = [c for c in (read_file(f) for f in files) if c]

    chain = SimpleSequentialChain(
        chains=[new_analyze_code_chain(openai_gpt35)], verbose=True
    )
    chain_results = chain.batch([{"input": code} for code in php_test_codes])

    results = [
        Result(f, c, r["output"])
        for f, c, r in zip(files, php_test_codes, chain_results)
    ]
    write_to_csv(results)
