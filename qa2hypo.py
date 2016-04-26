import re


# regex
regexs = {
    1: re.compile("^Where is (\w+)\?$"),
    2: re.compile("^Where is the (\w+)\?$"),
    3: re.compile("^Where was ([\w ]+) before the (\w+)\?$"),
    4.1: re.compile("^What is ([\w ]+) (\w+) of\?$"),
    4.2: re.compile("^What is (\w+) of the (\w+)\?$"),
    5.1: re.compile("^What did (\w+) give to (\w+)\?$"),
    5.2: re.compile("^Who gave the (\w+) to (\w+)\?$"),
    5.3: re.compile("^Who did (\w+) give the (\w+) to\?$"),
    5.4: re.compile("^Who gave the (\w+)\?$"),
    5.5: re.compile("^Who received the (\w+)\?$"),
    6: re.compile("Is (\w+) in the (\w+)\?$"),
    7: re.compile("^How many objects is (\w+) carrying\?$"),
    8: re.compile("^What is (\w+) carrying\?$"),
    # 9: same as 6
    # 10: same as 6
    # 11: same as 1
    # 12: same as 1
    # 13: same as 13
    # 14: same as 3
    15: re.compile("^What is (\w+) afraid of\?$"),
    16: re.compile("^What color is (\w+)\?$"),
    17: re.compile("^Is the ([\w+ ]+) (below|above|to the left of|to the right of) the ([\w ]+)\?$"),
    18.1: re.compile("^Is the ([\w ]+) (bigger|smaller) than the ([\w ]+)\?$"),
    18.2: re.compile("^Does the ([\w ]+) fit in the ([\w ]+)\?$"),
    19: re.compile("^How do you go from the (\w+) to the (\w+)\?$"),
    20.1: re.compile("^Where will (\w+) go\?$"),
    20.2: re.compile("^Why did (\w+) go to the (\w+)\?$"),
    20.3: re.compile("^Why did (\w+) get the (\w+)\?$"),
}


class C06(object):
    @staticmethod
    def format(containee, container, answer=None):
        if answer == "yes":
            return "{} is in the {}.".format(containee, container)
        elif answer == "no":
            return "{} is not in the {}.".format(containee, container)
        elif answer == "maybe":
            return "{} is maybe in the {}.".format(containee, container)
        raise Exception("Unrecognized answer: {}".format(answer))

class C08(object):
    @staticmethod
    def format(subject, answer=None):
        if answer == "nothing":
            return "{} is carrying nothing.".format(subject)
        else:
            return "{} is carrying the {answer}.".format(subject, answer=answer)


class C18_1(object):
    @staticmethod
    def format(subject, adjective, object, answer=None):
        if answer == "yes":
            return "The {} is {} than the {}.".format(subject, adjective, object)
        elif answer == "no":
            return "The {} is not {} than the {}.".format(subject, adjective, object)
        raise Exception("Unrecognized answer: {}".format(answer))


class C18_2(object):
    @staticmethod
    def format(subject, object, answer=None):
        if answer == "yes":
            return "The {} fits in the {}.".format(subject, object)
        elif answer == "no":
            return "The {} does not fit in the {}.".format(subject, object)
        raise Exception("Unrecognized answer: {}".format(answer))


out_strings = {
    1: "{0} is in the {answer}.",
    2: "The {0} is in the {answer}.",
    3: "The {0} is in the {answer}.",
    4.1: "The {0} is {1} of the {answer}.",
    4.2: "The {answer} is {0} of the {1}.",
    5.1: "{0} gave the {answer} to {1}.",
    5.2: "{answer} gave the {0} to {1}.",
    5.3: "{0} gave the {1} to {answer}.",
    5.4: "{0} gave the {answer}.",
    5.5: "{answer} received the {0}.",
    6: C06,
    7: "{0} is carrying {answer} objects.",
    8: C08,
    # 9: same as 6
    # 10: same as 6
    # 11: same as 1
    # 12: same as 1
    # 13: same as 13
    # 14: same as 3
    15: "{0} is afraid of {answer}.",
    16: "The color of {0} is {answer}.",
    17: "The {0} is {1} the {2}.",
    18.1: C18_1,
    18.2: C18_2,
    19: "You go {answer} from the {0} to the {1}.",
    20.1: "{0} will go to the {answer}.",
    20.2: "{0} went to the {1} because he is {answer}.",
    20.3: "{0} got the {1} because he is {answer}."

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
