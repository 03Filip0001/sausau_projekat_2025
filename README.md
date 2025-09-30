# Projekat IV: Telco Customer Churn Classification
Projekat rešava problem klasifikacije korisnika (`No Churn`/`Churn`) na osnovu priložene baze podataka o korisnicima

Dokumentacija projekta nalazi se u folderu [`documentation/SAUSAU projekat.pdf`](<documentation/SAUSAU projekat.pdf>) ovog repozitorijuma

## Pokretanje programa

Pozivom bilo koje `make` funkcionalnosti program kreira `python /.venv` direktorijum sa svim neophodnim bibliotekama za projekat

![make komanda za pokretanje](<documentation/pictures/make command.png>)

### Testiranje modela
Da bi se testirao neki model u terminalu je potrebno ukucati `make` bez ikakvih argumenata. Na osnovu [`congig.py`](config.py) fajla pokrece se jedan od 3 trenirana modela. Ako zelim da testiram drugi model i vidim njegove izlaze potrebno je da zakomentarisem trenutni model i odkomentarisem parametre za drugi model.

Bitni parametri za model: `DEFAULT_MODEL_CLASS`, `DEFAULT_MODEL_NAME`

![Primer dobrog podesavanja config.py fajla](<documentation/pictures/config models.png>)

### Treniranje modela
Da bi se trenirao model u terminalu je potrebno ukucati `make models`.
Program trenira sve modele u `MODEL_NAMES` promenljivi koja se nalazi u [`config.py`](<config.py>) prilikom funkcionalnosti treniranje modela. Parametre koje program testira se takodje nalaze u tom fajlu: `DEFAULT_PARAMS_LR`, `DEFAULT_PARAMS_RF`, `DEFAULT_PARAMS_GB`.

![Korisceni modeli](<documentation/pictures/models used.png>)

### Dodavanje novog modela
Dodavanje novog nepoznatog modela (model koji nije iz 3 koriscene klase) jos uvek nije omoguceno i program ce izbaciti gresku ako pokusamo da ga pokrenemo sa nepoznatim modelima

## Izlaz programa
Program nam daje osnove metrike modela kao sto su `preciznost`, `tacnost`, `odziv`, `F1-skor`, kao i `matricu konfuzije` i `klasifikacioni izvestaj`

![Izlaz modela](<documentation/pictures/model output.png>)