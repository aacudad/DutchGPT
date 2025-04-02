import json
import os
import time
import random
import hashlib
from typing import List, Dict
from pydantic import BaseModel, Field
from google import genai
import re
import uuid
from tqdm import tqdm

# Use a single API key from environment variable
client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))

# Define the data models for batch processing
class InstructionResponsePair(BaseModel):
    instruction: str
    response: str

class InstructionResponsePairs(BaseModel):
    pairs: List[InstructionResponsePair] = Field(..., description="List of instruction-response pairs")

# List of models to cycle through
models = [
    "gemini-2.0-flash-exp-image-generation",
    "gemini-2.0-flash",
    "gemini-2.0-flash-exp",
    "gemini-2.0-flash-lite-001",
    "gemini-2.0-pro-exp-02-05",
    "gemini-1.5-flash-001",
    "gemini-1.5-flash-002",
    "gemini-1.5-flash-8b",
    "gemini-1.5-pro",
    "gemini-2.0-flash-001"
]

# List of topics in Dutch
TOPICS = [
    "online winkelen", "webwinkel opzetten", "klantenservice", "productaanbevelingen",
    "reviews en beoordelingen", "kortingscodes zoeken", "retourbeleid", "bezorgopties",
    "duurzaam winkelen", "tweedehands aankopen",
    "technische ondersteuning", "smartphone problemen", "computerreparatie", "wifi-verbinding",
    "software-installatie", "smart home apparaten", "cloudopslag", "dataprivacy",
    "cybersecurity", "apparaat-synchronisatie",
    "hobbyprojecten", "fotografie tips", "muziekinstrumenten leren", "handwerken en breien",
    "verzamelingen beheren", "puzzels en spellen", "lezen en boekclubs", "tekenen en schilderen",
    "creatief schrijven", "modelbouw",
    "tuinieren", "woninginrichting", "huisrenovatie", "energiebesparing",
    "huishoudelijke apparaten", "schoonmaaktips", "binnenhuisarchitectuur", "klusjes in huis",
    "duurzaam wonen", "woningbeveiliging",
    "koken en recepten", "bakken", "dieetadvies", "vegetarisch koken",
    "wijnkennis", "restaurantadvies", "koffiebereiding", "voedselallergieën",
    "etiquette", "boodschappenlijsten",
    "gezondheid en welzijn", "fitnessadvies", "mentale gezondheid", "slaapproblemen",
    "meditatietechnieken", "stress verminderen", "voedingssupplementen", "yoga oefeningen",
    "alternatieve geneeswijzen", "griep en verkoudheid",
    "reisadviezen", "vakantiebestemmingen", "hotels boeken", "vliegtickets vinden",
    "openbaar vervoer", "autoverhuur", "backpacken", "stadswandelingen",
    "reisverzekeringen", "lokale gebruiken",
    "financieel advies", "spaarrekeningen", "hypotheekadvies", "belastingaangifte",
    "budgetteren", "pensioenplanning", "investeren voor beginners", "schulden aflossen",
    "cryptocurrency", "verzekeringen vergelijken",
    "opvoedingstips", "relatieproblemen", "familiebijeenkomsten", "dating advies",
    "cadeau-ideeën", "omgaan met pubers", "werk-privé balans", "communicatievaardigheden",
    "huwelijksplanning", "echtscheiding",
    "onderwijs en cursussen", "carrièreplanning", "sollicitatietips", "cv opstellen",
    "werkzoeken", "online leren", "thuiswerken", "bijscholing", "studiekeuze",
    "vakbondskwesties",
    "huisdierenverzorging", "dierentraining", "vogelspotten", "milieuproblematiek",
    "natuurfotografie", "dierenwelzijn", "wandelroutes", "visserij", "bosbouw", "stadsdieren",
    "sportactiviteiten", "hardlooptips", "fietshobby", "zwemtechnieken", "teamsporten",
    "buitenactiviteiten", "wintersport", "sportblessures", "sportuitrusting", "martiale arts",
    "entertainment opties", "filmrecensies", "tv-series", "concerten", "theatervoorstellingen",
    "podcasts", "gaming", "sociale media", "muziekstreaming", "kunsttentoonstellingen",
    "modestijl", "haarproducten", "huidverzorging", "make-up advies", "modetrends",
    "duurzame kleding", "accessoires", "seizoensmode", "persoonlijke stijl", "beautybehandelingen",
    "nieuwe gadgets", "smartphonekeuze", "laptopvergelijking", "draadloze oordopjes",
    "smartwatches", "e-readers", "digitale camera's", "virtual reality", "bluetooth-apparaten",
    "smart TV's",
    "actualiteiten", "lokale politiek", "klimaatverandering", "vrijwilligerswerk",
    "sociale kwesties", "verkiezingen", "buurtinitiatieven", "economische trends",
    "onderwijs hervormingen", "gezondheidszorgbeleid"
]
TOPICS += [
    # Technologie & Digitale Vaardigheden
    "robotica voor beginners", "AI-tools gebruiken", "cloud computing", "cyberpesten herkennen",
    "digitaal burgerschap", "toekomst van internet", "digitale geletterdheid", "digitale detox",
    "hoe werkt GPS", "hoe werken algoritmes", "nepnieuws herkennen", "computerbouw", 
    "toetsenbord shortcuts", "slimme wearables", "digitale veiligheid", "encryptie begrijpen",
    "informatie opzoeken", "open source software", "besturingssystemen vergelijken", 
    "digitaal minimalisme",

    # Creativiteit & Kunst
    "bullet journaling", "illustratie technieken", "abstract schilderen", "portret tekenen",
    "3D-ontwerpen", "kunstinstallaties", "muziekproductie", "korte verhalen schrijven",
    "digitale collage maken", "typografie ontwerpen", "boekbinden", "keramiek maken",
    "stoffen verven", "interactieve kunst", "stempels maken", "filmpjes monteren",
    "kleurenleer", "kunst analyseren", "korte film maken", "visual storytelling",

    # Levensstijl & Zelfontwikkeling
    "timemanagement", "gewoonten ontwikkelen", "minder smartphonegebruik", 
    "mentale veerkracht", "journaling", "beslissingen nemen", "effectief communiceren",
    "jezelf motiveren", "persoonlijke missie vinden", "carrièredoelen stellen",
    "burn-out voorkomen", "reflectie-oefeningen", "mindfulness in werk", 
    "eenvoudiger leven", "dagplanning", "jaardoelen", "visualisatie technieken",
    "omgaan met kritiek", "grenzen stellen", "echt luisteren",

    # Praktische Vaardigheden
    "band vervangen", "naaimachine gebruiken", "stopcontact aansluiten", 
    "schildertechnieken", "eenvoudige loodgieterklussen", "boormachine gebruiken", 
    "meubels opknappen", "plinten plaatsen", "kraan repareren", "kast in elkaar zetten",
    "muur isoleren", "gootsteen ontstoppen", "rolgordijnen ophangen", 
    "behang plakken", "vloer leggen", "planten verzorgen", "regenwater opvangen",
    "lamp ophangen", "veilig werken met stroom", "eigen compost maken",

    # Gezondheid & Vitaliteit
    "rugpijn voorkomen", "gezonde snacks", "sportoefeningen thuis", "water drinken", 
    "ademhalingsoefeningen", "koud douchen", "je houding verbeteren", "gezonde darmen", 
    "stoppen met roken", "ontbijten overslaan?", "voedsel bewaren", "warme-up oefeningen",
    "slaaphygiëne", "alcohol minderen", "luisteren naar je lichaam", 
    "natuurlijke remedies", "wandelen in de natuur", "daglicht en gezondheid", 
    "fit blijven op kantoor", "snel herstel bij griep",

    # Duurzaamheid & Milieu
    "duurzaam poetsen", "milieuvriendelijke kleding", "bewust watergebruik",
    "vleesconsumptie verminderen", "tweedehands kopen", "bomen planten", "plasticvrij leven",
    "eco-bewust reizen", "kringloopwinkels", "regenwater gebruiken", "zero waste koken",
    "lokaal voedsel kopen", "energiezuinige apparaten", "natuurlijke schoonmaakmiddelen", 
    "milieu-educatie", "hergebruik creatief", "klimaatactie ondernemen", 
    "composteren", "isoleren van je huis", "elektrisch rijden",

    # Onderwijs & Leren
    "studeren met flashcards", "concentratie verbeteren", "leren leren", "mindmaps maken",
    "examentraining", "leestempo verhogen", "zelfstudie tips", "online cursussen vinden", 
    "toetsangst overwinnen", "effectieve aantekeningen", "motivatie bij studeren", 
    "leren presenteren", "productief thuis studeren", "groepjeswerk aanpakken", 
    "plagiaat vermijden", "eigen leerstijl vinden", "digitale tools voor school", 
    "voorbereiden op tentamens", "taalvaardigheid verbeteren", "wetenschappelijke bronnen vinden",

    # Reizen & Ontdekken
    "roadtrip plannen", "kampeertips", "luchtvaartgeheimen", "stedentrips in Europa", 
    "off-the-grid reizen", "reisdagboek bijhouden", "werken vanuit het buitenland", 
    "rondreizen met trein", "taal leren op reis", "veilig reizen als vrouw",
    "duurzaam vliegen", "toeristenval vermijden", "lokale cultuur leren kennen", 
    "budgetreizen", "hostels boeken", "inpakken als minimalist", "reisvaccinaties", 
    "natuurparken bezoeken", "mooiste stranden", "wildlife spotten",

    # Maatschappij & Samenleving
    "burgerparticipatie", "democratie uitleggen", "verkiezingen volgen", 
    "betrokken buurtbewoner", "debat voeren", "fake news herkennen", 
    "inspraakmomenten", "vrijwilligerswerk doen", "maatschappelijke trends",
    "beïnvloeding in media", "sociale ongelijkheid", "klimaatdemonstraties", 
    "digitale rechten", "veilig internetgebruik", "online haat tegengaan", 
    "burgerinitiatieven", "inclusiviteit bevorderen", "leefbaarheid in de wijk", 
    "ken je grondrechten", "privacywetgeving begrijpen",

    # Extra niches & interesses
    "kruiden kweken", "sudoku strategieën", "zelf bier brouwen", "kaarsen maken", 
    "macramé knopen", "urban sketching", "eigen podcast starten", "kleding upcyclen", 
    "etherische oliën gebruiken", "boeken samenvatten", "interieurfotografie", 
    "dagelijkse affirmaties", "minimalistische kledingkast", "slow fashion", 
    "zelf thee mengen", "tiny houses ontwerpen", "virtuele musea bezoeken", 
    "taarten decoreren", "handlettering oefenen", "rituelen en routines creëren"
]

def generate_ir_pairs_batch(topics: List[str], model_name: str = "gemini-2.0-flash") -> List[Dict]:
    """Generate multiple instruction-response pairs for the given list of topics using Gemini API."""
    
    topics_str = ", ".join(topics)
    system_prompt = '''Je bent een assistent die helpt bij het genereren van Nederlandstalige instructiedata voor het trainen van AI-modellen.

Voor elk onderwerp dat je ontvangt, genereer je een natuurlijke instructie in het Nederlands, alsof een echte gebruiker iets wil leren, oplossen of verbeteren. Dit kan zowel in de vorm van een vraag als een directe opdracht.

Varieer actief in toon, stijl en formulering. Kies per onderwerp een passende vorm van instructie, zodat de dataset divers, realistisch en natuurlijk blijft.ieronder vind je 100 voorbeeldvormen. Deze zijn bedoeld als inspiratie — je bent volledig vrij om andere natuurlijke varianten te gebruiken.

---

🟦 **Voorbeelden van instructievragen (1–50):**

1. Hoe doe ik …?  
2. Wat is de beste manier om …?  
3. Maak een stappenplan voor …  
4. Geef tips over …  
5. Vat kort samen hoe je …  
6. Hoe pak ik … het beste aan?  
7. Wat moet ik doen als …?  
8. Leg uit hoe je …  
9. Hoe kan ik leren om …?  
10. Wat zijn de stappen om … te bereiken?  
11. Kun je uitleggen hoe … werkt?  
12. Hoe bereid ik me voor op …?  
13. Wat zijn veelgemaakte fouten bij …?  
14. Geef een overzicht van …  
15. Wat zijn de voordelen van …?  
16. Hoe werkt … precies?  
17. Wat heb ik nodig om te beginnen met …?  
18. Wat zijn de eerste stappen bij …?  
19. Wat zijn praktische tips voor …?  
20. Wat is belangrijk om te weten over …?  
21. Wat zijn de do's en don'ts van …?  
22. Hoe kan ik beter worden in …?  
23. Wat is een efficiënte manier om … te doen?  
24. Wat moet ik vermijden bij …?  
25. Welke tools kan ik gebruiken voor …?  
26. Hoe kan ik tijd besparen bij …?  
27. Kun je een gids geven voor …?  
28. Wat is het verschil tussen … en …?  
29. Geef een samenvatting van …  
30. Hoe pas ik … toe in de praktijk?  
31. Wat zijn handige technieken voor …?  
32. Hoe los ik problemen op met …?  
33. Hoe kan ik controleren of … goed werkt?  
34. Welke volgorde moet ik aanhouden bij …?  
35. Kun je een voorbeeld geven van …?  
36. Wat moet ik weten voordat ik begin met …?  
37. Hoe kan ik fouten vermijden tijdens …?  
38. Hoe lang duurt het om … te leren?  
39. Hoe test ik of … goed is uitgevoerd?  
40. Wat zijn de beste strategieën voor …?  
41. Welke voorbereiding is nodig voor …?  
42. Hoe maak ik een planning voor …?  
43. Wat moet ik doen na …?  
44. Hoe integreer ik … in mijn dagelijkse routine?  
45. Welke hulpmiddelen zijn nuttig voor …?  
46. Wat zijn valkuilen bij …?  
47. Hoe kan ik … optimaliseren?  
48. Hoe beoordeel ik of … succesvol is?  
49. Wat zijn alternatieven voor …?  
50. Hoe leer ik stap voor stap …?

---

🟩 **Voorbeelden van directe instructies (51–100):**

51. Maak een checklist over …  
52. Vat de belangrijkste punten van … samen  
53. Schrijf een korte uitleg over …  
54. Geef een concrete handleiding voor …  
55. Schrijf een stappenplan voor …  
56. Geef een voorbeeldsituatie van …  
57. Stel een lijst met tips op voor …  
58. Illustreer hoe … werkt  
59. Omschrijf het proces van …  
60. Maak een opsomming van belangrijke punten over …  
61. Bedenk een praktische aanpak voor …  
62. Presenteer een overzicht van …  
63. Formuleer duidelijke richtlijnen voor …  
64. Stel een stappenplan op voor …  
65. Ontwerp een korte cursus over …  
66. Geef een beknopte instructie voor …  
67. Leg in je eigen woorden uit hoe … werkt  
68. Vertel wat je moet weten over …  
69. Schrijf een tutorial over …  
70. Stel een beslisboom op voor …  
71. Ontwerp een checklist voor beginners over …  
72. Vat samen hoe je begint met …  
73. Schrijf een adviesrapport over …  
74. Geef een werkwijze voor …  
75. Maak een lijst met veelvoorkomende fouten bij …  
76. Schrijf een stappenplan met uitleg voor …  
77. Omschrijf een strategie voor …  
78. Presenteer een tip-overzicht voor …  
79. Ontwerp een voorbeeldschema voor …  
80. Formuleer succesfactoren van …  
81. Geef een concreet voorbeeld van …  
82. Stel een instructie op voor dagelijks gebruik van …  
83. Schrijf een gids voor beginners over …  
84. Stel een procesbeschrijving op van …  
85. Maak een overzicht van best practices rond …  
86. Geef een workflowvoorbeeld voor …  
87. Omschrijf het verschil tussen goed en fout gebruik van …  
88. Schrijf een stappenplan zonder jargon over …  
89. Leg op een eenvoudige manier uit hoe … werkt  
90. Maak een duidelijke uitleg voor kinderen over …  
91. Vertaal de theorie achter … naar de praktijk  
92. Verwerk een voorbeeldcase over …  
93. Stel een checklist op voor gevorderden over …  
94. Vat samen wat de risico's zijn van …  
95. Geef een visuele beschrijving van … (eventueel tekstueel)  
96. Bied een stappenplan met waarschuwingen voor …  
97. Geef instructies voor het oplossen van problemen met …  
98. Maak een plan van aanpak voor …  
99. Schrijf een informatieve tekst over …  
100. Beschrijf een scenario waarin … toegepast wordt

---

🔁 **Belangrijk:**  
Je hoeft je niet strikt aan deze voorbeelden te houden. Gebruik ze als inspiratie om telkens een **originele, natuurlijke en relevante instructie** te maken voor het onderwerp dat je krijgt.

📌 **Doel:**  
Na het genereren van de instructievraag of opdracht, geef je een **duidelijk, gedetailleerd en praktisch antwoord** dat geschikt is voor AI-training. Gebruik waar relevant opsommingen, voorbeelden, stappenplannen of heldere uitleg.
'''

    # Batch message to generate instruction-response pairs
    contents = f"{system_prompt}\n\nGenereer instructie-respons paren voor de volgende onderwerpen: {topics_str}. Voor elk onderwerp, genereer een natuurlijke instructie in het Nederlands (dit kan een vraag zijn zoals 'Hoe...' of een directe opdracht zoals 'Maak...'), gevolgd door een gedetailleerd en praktisch antwoord."
    
    batch_id = str(uuid.uuid4())[:8]
    max_retries = 3
    retry_delay = 10
    success = False
    batch_pairs = None
    model_index = models.index(model_name) if model_name in models else 0
    
    # Retry loop with model switching
    while not success:
        current_model = models[model_index]
        retries = 0
        
        while retries <= max_retries:
            try:
                print(f"Sending batch {batch_id} with {len(topics)} topics using model {current_model}")
                response = client.models.generate_content(
                    model=current_model,
                    contents=contents,
                    config={
                        'response_mime_type': 'application/json',
                        'response_schema': InstructionResponsePairs,
                    },
                )
                batch_pairs = response.parsed
                print(f"Successfully processed batch {batch_id}")
                success = True
                break
            except Exception as e:
                error_str = str(e)
                print(f"Error: {error_str[:150]}...")
                if '429' in error_str or 'RESOURCE_EXHAUSTED' in error_str:
                    if retries < max_retries:
                        print(f"Rate limit hit with model {current_model}. Retrying in {retry_delay}s [Attempt {retries+1}/{max_retries}]...")
                        time.sleep(retry_delay)
                        retries += 1
                    else:
                        break
                elif 'invalid JSON' in error_str or 'schema' in error_str:
                    try:
                        if hasattr(response, 'text'):
                            text_response = response.text
                            json_match = re.search(r'```json\s*([\s\S]*?)\s*```', text_response)
                            if json_match:
                                json_str = json_match.group(1)
                                batch_pairs = json.loads(json_str)
                                success = True
                                break
                    except Exception:
                        pass
                    if retries < max_retries:
                        print(f"JSON error. Retrying with model {current_model} [Attempt {retries+1}/{max_retries}]...")
                        time.sleep(retry_delay)
                        retries += 1
                    else:
                        break
                else:
                    break
        
        if not success:
            model_index = (model_index + 1) % len(models)
            print(f"Switching to model: {models[model_index]}")
    
    if success and batch_pairs:
        if hasattr(batch_pairs, 'dict'):
            # If we got a parsed Pydantic object
            return [pair.dict() for pair in batch_pairs.pairs]
        elif isinstance(batch_pairs, dict) and 'pairs' in batch_pairs:
            # If we got a dictionary with 'pairs' key
            return batch_pairs['pairs']
        else:
            # Fallback if the structure is unexpected
            print("Warning: Unexpected response format from Gemini.")
            return []
    else:
        return []

# Append multiple entries to a JSONL file
def append_to_jsonl(entries: List[Dict], filename: str):
    """Append multiple entries to the JSONL file."""
    with open(filename, 'a', encoding='utf-8') as f:
        for entry in entries:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')

# Create the dataset with instruction-response pairs in batches
def create_dataset(num_samples: int, batch_size: int = 5, model_name: str = "gemini-2.0-flash") -> List[Dict]:
    """Generate a dataset of instruction-response pairs in batches and save them to a JSONL file."""
    dataset = []
    jsonl_filename = "dutch_ir_pairs_gemini_1.jsonl"
    
    total_generated = 0
    total_batches = (num_samples + batch_size - 1) // batch_size  # Ceiling division
    
    with tqdm(total=num_samples, desc="Generating pairs") as pbar:
        while total_generated < num_samples:
            current_batch_size = min(batch_size, num_samples - total_generated)
            batch_topics = [random.choice(TOPICS) for _ in range(current_batch_size)]
            print(f"\nGenerating batch {total_generated//batch_size + 1}/{total_batches} with {current_batch_size} topics (Total so far: {total_generated}/{num_samples})...")
            
            ir_pairs = generate_ir_pairs_batch(batch_topics, model_name)
            
            if not ir_pairs:
                print("Failed to generate pairs for this batch. Retrying with new topics...")
                time.sleep(5)
                continue
            
            entries = []
            for ir_data in ir_pairs:
                instruction = ir_data["instruction"]
                response = ir_data["response"]
                conversations = [
                    {"role": "user", "content": instruction},
                    {"role": "assistant", "content": response}
                ]
                hash_value = hashlib.sha256(instruction.encode('utf-8')).hexdigest()
                entry = {
                    "content": instruction,
                    "hash": hash_value,
                    "conversations": conversations
                }
                entries.append(entry)
                dataset.append(entry)
            
            append_to_jsonl(entries, jsonl_filename)
            total_generated += len(entries)
            pbar.update(len(entries))
            print(f"Successfully generated and appended batch of {len(entries)} pairs. Total: {total_generated}/{num_samples}")
            
            # Save a backup every 100 samples
            if total_generated % 100 < batch_size:
                backup_filename = f"dutch_ir_pairs_gemini_backup_{total_generated}.jsonl"
                with open(backup_filename, 'w', encoding='utf-8') as f:
                    for entry in dataset:
                        f.write(json.dumps(entry, ensure_ascii=False) + '\n')
                print(f"Backup saved to {backup_filename}")
            
            time.sleep(2)  # Avoid hitting rate limits
    
    return dataset

# Format the dataset for Hugging Face
def format_for_huggingface(dataset: List[Dict], filename: str = "dutch_ir_pairs_huggingface_gemini_1.json"):
    """Format the dataset for Hugging Face compatibility."""
    formatted_data = [{"content": entry["content"], "hash": entry["hash"], "conversations": entry["conversations"]} for entry in dataset]
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(formatted_data, f, ensure_ascii=False, indent=2)
    print(f"Formatted dataset saved to {filename}")

# Main execution function
def main():
    num_samples = 10000  # Adjust the number of samples as needed
    batch_size = 5       # Set batch size here
    model_name = "gemini-2.0-flash"  # Starting model
    
    print("Starting Dutch instruction-response pair dataset generation using Gemini...")
    print(f"Will generate {num_samples} pairs in batches of {batch_size}")
    print(f"Using {len(TOPICS)} different topics")
    print(f"Starting model: {model_name}")
    
    # Create output directory if it doesn't exist
    os.makedirs("outputs", exist_ok=True)
    
    dataset = create_dataset(num_samples, batch_size, model_name)
    
    if not dataset:
        print("No pairs were successfully generated.")
        return
    
    format_for_huggingface(dataset)
    
    print(f"Generated {len(dataset)} pairs successfully.")
    
    if dataset:
        print("\nSample instruction-response pair:")
        sample = dataset[0]
        print(f"Instruction: {sample['content']}")
        print("Conversation:")
        for idx, msg in enumerate(sample['conversations']):
            print(f"{idx+1}. {msg['role']}: {msg['content']}")

if __name__ == "__main__":
    main()