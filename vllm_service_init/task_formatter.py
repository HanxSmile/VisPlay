class TaskFormatter:
    ALL_CHOICES = "ABCDEFG"

    def single_choice(self, response):
        response = response.split("<answer>")[-1].split("</answer>")[0].strip()
        if response in self.ALL_CHOICES:
            return response
        return ""

    def multi_choice(self, response):
        response = response.split("<answer>")[-1].split("</answer>")[0]
        choices = response.strip().split(",")
        choices = [_.strip() for _ in choices]
        for choice in choices:
            if choice not in self.ALL_CHOICES:
                return tuple()
        return tuple(sorted(choices))

    def closed_ended(self, response):
        response = response.split("<answer>")[-1].split("</answer>")[0].strip().lower()
        if "yes" in response or "是" in response:
            return True
        if "no" in response or "否" in response:
            return False
        return False

    def __call__(self, q_type, response):
        return getattr(self, q_type)(response)
