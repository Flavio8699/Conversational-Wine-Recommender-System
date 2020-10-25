# This files contains your custom actions which can be used to run
# custom Python code.
#
# See this guide on how to implement these action:
# https://rasa.com/docs/rasa/core/actions/#custom-actions/


# This is a simple example for a custom action which utters "Hello World!"

import sqlite3
from typing import Any, Text, Dict, List

from rasa_sdk import Action, Tracker
from rasa_sdk.executor import CollectingDispatcher


class ActionAskInfo(Action):

    def name(self) -> Text:
        return "action_ask_info"

    def run(self, dispatcher, tracker, domain):
        cursor = self.connection()
        value = tracker.latest_message['entities'][0]['value']

        cursor.execute(
            'SELECT Summary FROM reviews WHERE UPPER(Wine) = ? COLLATE NOCASE', (value.upper(),))
        result = cursor.fetchone()

        if result is None:
            dispatcher.utter_message(
                "I am sorry, we didn't find anything according to your search")
            cursor.execute(
                'SELECT Wine FROM reviews WHERE UPPER(Wine) LIKE ? ORDER BY RANDOM() LIMIT 5 COLLATE NOCASE', ("%"+value.upper()+"%",))
            results = cursor.fetchall()

            if len(results) > 0:
                dispatcher.utter_message("Did you maybe mean one of these:")
                for row in results:
                    dispatcher.utter_message(row[0])
        else:
            dispatcher.utter_message(result[0])
        return []

    def connection(self):
        conn = None
        try:
            conn = sqlite3.connect('data.sqlite').cursor()
        except Error as e:
            print(e)

        return conn
