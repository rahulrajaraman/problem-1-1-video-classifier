"""Visual text prompts for each Video-MME domain (via its sub-categories)."""

from __future__ import annotations

SUBCATEGORIES = {
    "Artistic Performance": {
        "Acrobatics": "acrobats performing tricks",
        "Magic Show": "a magician performing a magic trick for an audience",
        "Stage Play": "actors performing a play on a theatre stage",
        "Variety Show": "a TV variety show with a host and performers on stage",
    },
    "Film & Television": {
        "Animation": "an animated cartoon",
        "Documentary": "a nature or history documentary",
        "Movie & TV Show": "a scene from a movie or TV series",
        "News Report": "a TV news broadcast with a news anchor",
    },
    "Knowledge": {
        "Astronomy": "an astronomy video about planets, stars and space",
        "Biology & Medicine": "an educational video about biology or medicine",
        "Finance & Commerce": "a video explaining finance, business or the economy",
        "Geography": "an educational video about geography and maps",
        "Humanity & History": "a history lecture with old pictures and maps",
        "Life Tip": "a how-to tutorial with practical tips",
        "Literature & Art": "a lecture about literature or art",
        "Technology": "a video explaining technology or gadgets",
    },
    "Life Record": {
        "Daily Life": "a vlog of someone's daily life",
        "Exercise": "a person doing a workout",
        "Fashion": "a fashion vlog showing outfits",
        "Food": "a person cooking or eating food",
        "Handicraft": "a person making handicrafts by hand",
        "Travel": "a travel vlog visiting places",
    },
    "Sports Competition": {
        "Athletics": "an athletics race on a running track",
        "Basketball": "a basketball game",
        "Esports": "a competitive video game esports match",
        "Football": "a football match in a stadium",
        "Other Sports": "a sports competition",
    },
}

TEMPLATES = ["{}", "a video frame of {}.", "a screenshot of {}."]
