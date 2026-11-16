# -*- coding: iso-8859-1 -*-

TOKENS = ["<blank>", "zwei", "drei", "vier", "fünf", "sechs", "sieben", "acht", "neun",
          "eo", "ba", "ndb", "het", "hettox", "hetstab", "hetmet",
          "moly", "mo", "ly", "lyklein", "lymittel", "lygross", "lymag"]

token_to_id = {token: i for i, token in enumerate(TOKENS)}
id_to_token = {i: token for i, token in enumerate(TOKENS)}