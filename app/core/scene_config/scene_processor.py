class SceneProcessor:
    def process(self, user_input, context, input_object):
        """
        Processes the user input and returns the next scene to be executed.
        :param input_object:
        :param user_input: The user input as a string.
        :param context: The context object containing the current state of the conversation.
        :return: The name of the next scene to be executed.
        """
        raise NotImplementedError
