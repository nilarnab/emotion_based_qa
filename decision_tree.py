"""
The following decisions can be taken
1. Mood of the room (joyful, sorrowful, neutral)
2. Total number of people 
3. Is loneliness there
4. How many people are happy, sad, surprised
"""

from typing_extensions import Self
from xmlrpc.client import TRANSPORT_ERROR


ALL_MOOD = 0.9
MOSTLY_MOOD = 0.5
SOME_PEOPLE_MOOD = 0.2

class DecisionMaker:
    def __init__(self, moods):
        self.face_num = sum(moods.values())
        self.moods = moods
    
    def aggregate_mood_finder(self):

        """
        adj: everyone is, mostly, 

        param: if any mood gets over 90 %  then that mood is the mood of the room
        """

        context = []

        moods = []
        
        for mood in self.moods:
            moods.append([self.moods[mood], mood])
        
        moods.sort(reverse=True)
        
        # generating general mood based context
        for mood_pair in moods:
            if mood_pair[0] > 1:
                context.append(str(mood_pair[0]) + " people are " + mood_pair[1])
            elif mood_pair[0] == 1:
                context.append(str(mood_pair[0]) + " person is " + mood_pair[1])

        # generating rule based mood
        if moods[0][0] / self.face_num > ALL_MOOD:
            if self.face_num > 1:
                context.append('Everyone is ' + moods[0][1])
        elif moods[0][0] / self.face_num > MOSTLY_MOOD:
            if self.face_num > 1:
                context.append('Everyone is mostly ' + moods[0][1])
        else:
            if self.face_num > 1:
                context.append('People are having mixed moods')

        # generating rule based mood
        if moods[1][0] / self.face_num > SOME_PEOPLE_MOOD:
            context.append('Some people are ' + moods[1][1])



        # Generating complex emotions

        return context

    def find_mood_lonliness(self):
        if self.face_num == 1:
            if sum([self.moods['happy'] + self.moods['surprised'] + self.moods['surprised']]) == 0:
                return ['There is lonliness in the picture']

        return []


    def find_emotional_caption(self):

        captions = []

        for context in self.aggregate_mood_finder():
            captions.append(context)

        for context in self.find_mood_lonliness():
            captions.append(context)

        return captions


if __name__ == '__main__':
    moods = {
        'disgusted': 0,
        'angry': 0,
        "fearful": 1,
        "happy": 15,
        "neutral": 10,
        "sad": 0,
        "suprised": 0 
    }

    decisionMaker = DecisionMaker(moods)

    res = decisionMaker.aggregate_mood_finder()

    print(res)
    










