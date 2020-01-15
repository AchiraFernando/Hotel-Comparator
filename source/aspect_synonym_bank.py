from nltk.corpus import wordnet

class AspectSynonymBank:

    location = ['location', 'placement', 'locating', 'position', 'positioning', 'emplacement', 'localization',
                'localisation', 'fix', 'ocean', 'sea', 'sunset', 'sundown', 'wave', 'beautiful', 'colonial',
                'atmosphere', 'ambiance', 'air', 'aura', 'building', 'construction', 'construct', 'progress',
                'establish', 'place', 'spot', 'property', 'home', 'situation', 'situated', 'station', 'train', 'road',
                'city', 'countryside', 'urban', 'rural', 'identify', 'localize', 'view', 'perspective', 'scene', 'vista',
                'panorama', 'scenery', 'horizon', 'watch', 'heritage', 'inheritance', 'beach', 'traditional', 'elegant',
                'graceful', 'refined', 'charm', 'charming', 'unattractive', 'untempting', 'authentic', 'landscape']

    food = ['food', 'nutrient', 'breakfast', 'drink', 'drinking', 'booze', 'beverage', 'dinner', 'restaurant', 'bar',
            'liquor', 'buffet', 'snack_bar', 'wine', 'lunch', 'meal', 'menu', 'cook', 'prepare', 'cake', 'fruit',
            'desert', 'authentic']

    facilities = ['facility', 'installation', 'facility', 'readiness', 'construction', 'building', 'make', 'progress',
                  'establish', 'swimming', 'swim', 'pool', 'pond', 'ac', 'air conditioner', 'historic', 'historical',
                  'lobby', 'hall', 'accommodation', 'accommodate', 'spa', 'health spa', 'gymnasium', 'gym', 'clean',
                  'renovation', 'redevelopment', 'restoration', 'refurbishment', 'massage', 'comfortable', 'comfy',
                  'well fixed', 'experience', 'feel', 'sport', 'athletics', 'play', 'game', 'pool table', 'parking',
                  'carlot', 'jacuzzi', 'steambath']

    service = ['help', 'serve', 'serving', 'service', 'help', 'servicing', 'serve', 'lovingness', 'caring', 'care',
               'manage', 'handle', 'worry', 'helpful', 'sensational', 'stunning', 'complementary', 'complemental',
               'ascent', 'upgrade', 'promote', 'advance', 'reception', 'receipt', 'response', 'experience', 'receive'
               , 'feel', 'unattractive', 'untempting', 'uncomfortable', 'hospitality', 'family', 'home', 'stay', 'rest'
               , 'delay', 'quality', 'tone', 'valet', 'free', 'class', 'category', 'grade', 'classify']

    room = ['way', 'room', 'board', 'balcony', 'view', 'position', 'aspect', 'scene', 'window', 'bedroom', 'chamber',
            'bathroom', 'bath', 'toilet', 'sleep', 'nap', 'rest', 'accommodation', 'comfortable', 'comfy',
            'unattractive', 'untempting', 'uncomfortable', 'bed', 'washroom', 'interior']

    staff = ['staff', 'faculty', 'friendly', 'helpful', 'management', 'attentive', 'thoughtful', 'attention', 'manager',
             'receptionist', 'office', 'reception']

    value = ['price', 'value', 'treasure', 'appreciate', 'respect', 'measure', 'evaluate', 'valuate', 'assess',
             'appraise', 'rate', 'costly', 'pricey', 'pricy', 'expensive', 'cost', 'toll', 'budget', 'pay', 'paid'
             , 'paying', 'payment', 'wage', 'earnings', 'bill', 'account', 'cheque', 'invoice', 'tax', 'taxation']

    cleanliness = ['cleanliness', 'clean', 'cleanse' 'neat', 'clear', 'fresh', 'scent', 'uninfected', 'fair', 'white',
                   'infected', 'dirty', 'gross']

    def get_synonyms_for_aspect(self, aspect):

        if aspect == 'location':
            return self.location
        elif aspect == 'food':
            return self.food
        elif aspect == 'facilities':
            return self.facilities
        elif aspect == 'service':
            return self.service
        elif aspect == 'room':
            return self.room
        elif aspect == 'staff':
            return self.staff
        elif aspect == 'value':
            return self.value
        elif aspect == 'cleanliness':
            return self.cleanliness
        else:
            synonyms = []
            for syn in wordnet.synsets(aspect):
                for lm in syn.lemmas():
                    synonyms.append(lm.name())
            return synonyms

