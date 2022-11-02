from unittest import TestCase

from utils import split_on_speaker_change


class Test(TestCase):
    def test_split_on_speaker_change(self):
        text = """
        {taras_sereda}
        November 1, nice sunny day in Kyiv - capital of brave and resilient people.
        {joe_rogan}
        Talking, here again
        {taras_sereda}
        Another view on that problem
        {joe_rogan}
        see you in Cali!
        """
        res = split_on_speaker_change(text)
        res = list(res)
        self.assertEquals(res[0][0].strip(),
                          "taras_sereda")
        self.assertEquals(res[0][1].strip(),
                          "November 1, nice sunny day in Kyiv - capital of brave and resilient people.")
        self.assertEquals(res[2][0].strip(),
                          "taras_sereda")
        self.assertEquals(res[2][1].strip(),
                          "Another view on that problem")
        self.assertEquals(res[3][0].strip(),
                          "joe_rogan")
        self.assertEquals(res[3][1].strip(),
                          "see you in Cali!")

