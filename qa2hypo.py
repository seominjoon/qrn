import re


# regex
regex_where = re.compile("^Where is ([\w+ ?]+)\?$")
target_where = "{0} is in the {answer}."
def _where(question, answer):
    result = regex_where.match(question)
    if result:
        return target_where.format(*result.groups(), answer=answer).capitalize()
    return result


def qa2hypo(question, answer):
    question = question.lstrip().rstrip()
    answer = answer.lstrip().rstrip()
    where = _where(question, answer)
    if where:
        return where
    raise Exception("Unknown question format: {}".format(question))


def main():
    question = "Where is Mary?"
    answer = "office"
    print(qa2hypo(question, answer))

if __name__ == "__main__":
    main()
