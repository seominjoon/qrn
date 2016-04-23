import re


# regex
regexs = {
    1: re.compile(r"^Where is ([\w+ ?]+)\?$"),
    # 2: same as 1
    3: re.compile("^Where was ([\w+ ?]+) before ([\w+ ?]+)\?$"),
    4.1: re.compile("^What is ([\w+ ?]+) (\w+) of\?$"),
    4.2: re.compile("^What is (\w+) of ([\w+ ?]+)\?$"),
    5.1: re.compile("^What did (\w+) give to (\w+)\?$"),
    5.2: re.compile("^Who gave ([\w+ ?]+) to (\w+)\?$"),
    5.3: re.compile("^Who did (\w+) give ([\w+ ?]+) to\?$"),
    5.4: re.compile("^Who gave ([\w+ ?]+)\?$"),
    5.5: re.compile("^Who received ([\w+ ?]+)\?$"),
    6: re.compile("^Is (\w+) in ([\w+ ?]+)\?$"),
}

class C06(object):
    @staticmethod
    def format(containee, container, answer=None):
        if answer == "yes":
            return "{} is in {}.".format(containee, container)
        elif answer == "no":
            return "{} is not in {}.".format(containee, container)
        raise Exception("Unrecognized answer: {}".format(answer))

out_strings = {
    1: "{0} is in the {answer}.",
    3: "{0} is in the {answer}.",
    4.1: "{0} is {1} of {answer}.",
    4.2: "{answer} is {0} of {1}.",
    5.1: "{0} gave {answer} to {1}.",
    5.2: "{answer} gave {0} to {1}.",
    5.3: "{0} gave {1} to {answer}.",
    5.4: "{0} gave {answer}.",
    5.5: "{answer} received {0}.",
    6: C06,
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
