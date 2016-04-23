import re


# regex
regexs = {
    1: re.compile(r"^Where is ([\w+ ?]+)\?$"),
    # 2: same as 1
    3: re.compile("^Where was ([\w+ ?]+) before ([\w+ ?]+)\?$"),
    4.1: re.compile("^What is ([\w+ ?]+) (\w+) of\?$"),
    4.2: re.compile("^What is (\w+) of ([\w+ ?]+)\?$")
}

out_strings = {
    1: "{0} is in the {answer}.",
    3: "{0} is in the {answer}.",
    4.1: "{0} is {1} of {answer}.",
    4.2: "{answer} is {0} of {1}.",
}


def apply(regex, string, question, answer):
    result = regex.match(question)
    if result:
        return string.format(*result.groups(), answer=answer).capitalize()
    return result


def qa2hypo(question, answer):
    question = question.lstrip().rstrip()
    answer = answer.lstrip().rstrip()
    for task, regex in regexs.items():
        string = out_strings[task]
        result = apply(regex, string, question, answer)
        if result:
            return result
    raise Exception("Unknown question format: {}".format(question))


def main():
    question = "Where is Mary?"
    answer = "office"
    print(qa2hypo(question, answer))

if __name__ == "__main__":
    main()
