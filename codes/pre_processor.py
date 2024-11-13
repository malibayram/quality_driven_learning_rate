import pandas as pd

speciality_mapping = {
    "acil-tip": ["acil-tip"],
    "akupunktur": ["akupunktur", "akupunktur-biorezonans"],
    "algoloji": ["algoloji", "fiziksel-tip-ve-rehabilitasyon-fiziksel-tip-ve-rehabilitasyon-algoloji", "noroloji-noroloji-algoloji"],
    "anatomi": ["anatomi"],
    "anestezi-ve-reanimasyon": ["anestezi-ve-reanimasyon", "anestezi-ve-reanimasyon-algoloji-anestezi-ve-reanimasyon"],
    "aile-hekimligi": ["aile-hekimligi", "pratisyen-hekimlik"],
    "aile-danismani": ["aile-danismani", "psikoloji-aile-danismani-psikolog", "klinik-psikolog-psikoloji-aile-danismani-psikolog", "dil-konusma-bozuklari-b-aile-danismani"],
    "diyetisyen": ["diyetisyen"],
    "beyin-ve-sinir-cerrahisi": ["beyin-ve-sinir-cerrahisi"],
    "cocuk-cerrahisi": ["cocuk-cerrahisi", "cocuk-cerrahisi-cocuk-cerrahisi-cocuk-urolojisi", "cocuk-cerrahisi-cocuk-cerrahisi-cocuk-urolojisi-cocuk-urolojisi-uroloji"],
    "cocuk-sagligi-ve-hastaliklari": ["cocuk-sagligi-ve-hastaliklari", "cocuk-sagligi-ve-hastaliklari-cocuk-enfeksiyon-hastaliklari", 
                                      "cocuk-sagligi-ve-hastaliklari-cocuk-endokrinoloji", "cocuk-sagligi-ve-hastaliklari-cocuk-gastroenterolojisi", 
                                      "cocuk-sagligi-ve-hastaliklari-cocuk-genetik-hastaliklari", "cocuk-sagligi-ve-hastaliklari-cocuk-hematoloji", 
                                      "cocuk-sagligi-ve-hastaliklari-cocuk-kardiyolojisi", "cocuk-sagligi-ve-hastaliklari-cocuk-nefrolojisi", 
                                      "cocuk-sagligi-ve-hastaliklari-cocuk-norolojisi", "cocuk-sagligi-ve-hastaliklari-cocuk-psikiyatrisi", 
                                      "cocuk-sagligi-ve-hastaliklari-cocuk-romatolojisi", "cocuk-sagligi-ve-hastaliklari-neonatoloji", 
                                      "cocuk-sagligi-ve-hastaliklari-cocuk-gogus-hastaliklari", "cocuk-sagligi-ve-hastaliklari-cocuk-immunolojisi-ve-alerji-hastaliklari"],
    "dermatoloji": ["dermatoloji"],
    "dis-hekimi": ["dis-hekimi", "dis-hekimi-pedodonti", "dis-hekimi-ortodonti", "dis-hekimi-oral-implantoloji", 
                  "dis-hekimi-periodontoloji", "dis-hekimi-restoratif-dis-tedavisi", "dis-hekimi-endodonti", 
                  "dis-hekimi-agiz-dis-ve-cene-cerrahisi", "dis-hekimi-dis-protezi-uzmani", "dis-hekimi-oral-diagnoz-ve-radyoloji",
                  "dis-hekimi-koruyucu-dis-hekimlig"],
    "endokrinoloji-ve-metabolizma-hastaliklari": ["dahiliye-ve-ic-hastaliklari-endokrinoloji-ve-metabolizma-hastaliklari", "endokrinoloji-ve-metabolizma-hastaliklari"],
    "enfeksiyon-hastaliklari-ve-klinik-mikrobiyoloji": ["enfeksiyon-hastaliklari-ve-klinik-mikrobiyoloji"],
    "fiziksel-tip-ve-rehabilitasyon": ["fiziksel-tip-ve-rehabilitasyon", "fiziksel-tip-ve-rehabilitasyon-fiziksel-tip-ve-rehabilitasyon-romatoloji"],
    "genel-cerrahi": ["genel-cerrahi", "genel-cerrahi-cerrahi-onkoloji", "genel-cerrahi-gastroenteroloji-cerrahisi", "genel-cerrahi-endokrin-cerrahisi", "genel-cerrahi-meme-cerrahisi", "genel-cerrahi-proktoloji"],
    "gogus-cerrahisi": ["gogus-cerrahisi"],
    "gogus-hastaliklari": ["gogus-hastaliklari", "gogus-hastaliklari-alerji-ve-gogus-hastaliklari"],
    "gastroenteroloji": ["dahiliye-ve-ic-hastaliklari-gastroenteroloji", "gastroenteroloji"],
    "histoloji-ve-embriyoloji": ["histoloji-ve-embriyoloji"],
    "immunoloji": ["dahiliye-ve-ic-hastaliklari-immunoloji-ve-alerji-hastaliklari", "dahiliye-ve-ic-hastaliklari-immunoloji"],
    "kadin-hastaliklari-ve-dogum": ["kadin-hastaliklari-ve-dogum", "kadin-hastaliklari-ve-dogum-jinekolojik-onkoloji", 
                                    "kadin-hastaliklari-ve-dogum-ureme-endokrinolojisi-ve-infertilite", "kadin-hastaliklari-ve-dogum-perinatoloji"],
    "kardiyoloji": ["kardiyoloji"],
    "kalp-damar-cerrahisi": ["kalp-damar-cerrahisi", "kalp-damar-cerrahisi-damar-cerrahisi"],
    "kulak-burun-bogaz-hastaliklari": ["kulak-burun-bogaz-hastaliklari"],
    "noroloji": ["noroloji", "noroloji-klinik-norofizyoloji"],
    "nukleer-tip": ["nukleer-tip"],
    "plastik-rekonstruktif-ve-estetik-cerrahi": ["plastik-rekonstruktif-ve-estetik-cerrahi", "ortopedi-ve-travmatoloji-el-cerrahisi-ve-mikrocerrahi",
                                                "plastik-rekonstruktif-ve-estetik-cerrahi-el-cerrahisi-ve-mikrocerrahi-plastik"],
    "psikiyatri": ["psikiyatri"],
    "psikoloji": ["psikoloji", "psikolojik-danisman", "psikolojik-danisman-psikoloji", "psikoterapi", "psikoloji-psikoonkoloji"],
    "radyoloji": ["radyoloji", "radyoloji-girisimsel-radyoloji", "radyoloji-uroradyoloji"],
    "romatoloji": ["romatoloji-dahiliye-ve-ic-hastaliklari"],
    "spor-hekimligi": ["spor-hekimligi"],
    "tibbi-genetik": ["tibbi-genetik"],
    "tibbi-onkoloji": ["tibbi-onkoloji", "tibbi-onkoloji-dahiliye-ve-ic-hastaliklari"],
    "tibbi-biyokimya": ["tibbi-biyokimya"],
    "tibbi-patoloji": ["tibbi-patoloji"],
    "uroloji": ["uroloji", "uroloji-androloji"],
    "veteriner": ["veteriner"]
}

specialty_mapping2 = {
    "cerrahi": [
        "ortopedi-ve-travmatoloji", "genel-cerrahi", "beyin-ve-sinir-cerrahisi",
        "gogus-cerrahisi", "kalp-damar-cerrahisi", "cocuk-cerrahisi", 
        "plastik-rekonstruktif-ve-estetik-cerrahi", "uroloji"
    ],
    "kadın-dogum": [
        "kadin-hastaliklari-ve-dogum", "cocuk-sagligi-ve-hastaliklari", 
        "cocuk-psikiyatrisi", "aile-hekimligi"
    ],
    "dahiliye": [
        "dahiliye-ve-ic-hastaliklari", "dahiliye-ve-ic-hastaliklari-nefroloji",
        "endokrinoloji-ve-metabolizma-hastaliklari", "gastroenteroloji", 
        "immunoloji", "romatoloji", "dahiliye-ve-ic-hastaliklari-geriatri",
        "dahiliye-ve-ic-hastaliklari-hematoloji"
    ],
    "psikiyatri-ve-psikoloji": [
        "psikiyatri", "psikoloji", "klinik-psikolog", "klinik-psikolog-psikoloji",
        "aile-danismani", "pedagoji"
    ],
    "diş-hekimliği": ["dis-hekimi", "pedodonti"],
    "dermatoloji": ["dermatoloji", "medikal-estetik-sertifikali-tip-doktoru"],
    "göz": ["goz-hastaliklari"],
    "radyoloji-ve-onkoloji": ["radyoloji", "radyasyon-onkolojisi", "tibbi-onkoloji", "nukleer-tip"],
    "fizik-tedavi-ve-rehabilitasyon": ["fiziksel-tip-ve-rehabilitasyon", "fizyoterapi", "ergoterapi"],
    "spor-hekimliği": ["spor-hekimligi"],
    "anestezi-ve-algoloji": ["anestezi-ve-reanimasyon", "algoloji"],
    "genetik": ["tibbi-genetik"],
    "farmakoloji": ["tibbi-farmakoloji"],
    "biyokimya": ["tibbi-biyokimya"],
    "patoloji": ["tibbi-patoloji"],
    "enfeksiyon-ve-immunoloji": ["enfeksiyon-hastaliklari-ve-klinik-mikrobiyoloji", "immunoloji"],
    "geleneksel-tamamlayici-tip": ["geleneksel-ve-tamamlayici-tip", "akupunktur"],
    "çocuk-gelişimi": ["cocuk-gelisim-uzmani"],
    "konuşma-terapisi": ["dil-konusma-bozuklari-b"],
    "anatomi-ve-histoloji": ["anatomi", "histoloji-ve-embriyoloji"],
    "veterinerlik": ["veteriner"],
    "kardiyoloji": ["kardiyoloji"],
    "göğüs-hastaliklari": ["gogus-hastaliklari"],
    "kulak-burun-bogaz": ["kulak-burun-bogaz-hastaliklari"]
}

title_mapping = {
    "dr-ogr-uyesi": ["Dr. Öğr. Üyesi", "Dr.Öğr.Üyesi", "Dr. Öğr. Üyesi Dt."],
    "uzm-dr": ["Uzm. Dr.", "Uzm. Dr. Dt."],
    "op-dr": ["Op. Dr."],
    "dyt": ["Dyt.", "Dr. Dyt.", "Uzm. Dyt."],
    "yrd-doc-dr": ["Yrd. Doç. Dr."],
    "doc-dr": ["Doç. Dr.", "Doç. Dr. Dt.", "Doç. Dr. Psk. Dan"],
    "prof-dr": ["Prof. Dr.", "Prof. Dr. Dt."],
    "dr": ["Dr."],
    "uzm-kl-psk": ["Uzm. Kl. Psk."],
    "dt": ["Dt.", "Dr. Dt.", "Uzm. Dt."],
    "veteriner-hekim": ["Veteriner Hekim", "Vet."],
    "uzm-psk": ["Uzm. Psk.", "Uzm. Psk. Dan."],
    "psk": ["Psk.", "Psikoterapist", "Klinik Psikolog ", "Dr. Psk.", "Dr. Psk. Dan."],
    "psk-dan": ["Psk. Dan.", "Aile Danışmanı"],
    "fzt": ["Fzt."],
    "ergoterapist": ["Ergoterapist"],
    "cocuk-gelisim-uzmani": ["Çocuk Gelişim Uzmanı"],
    "dil-ve-konusma-terapisti": ["Dil ve Konuşma Terapisti"],
    "pedagog": ["Pedagog"]
}

title_mapping2 = {
    "profesor": ["prof-dr"],
    "docent": ["doc-dr"],
    "dr-ogr-uyesi": ["dr-ogr-uyesi", "yrd-doc-dr"],
    "uzman-doktor": ["uzm-dr", "op-dr"],
    "diyetisyen": ["dyt"],
    "dr-diyetisyen": ["Dr. Dyt."],
    "psikolog": ["psk", "psk-dan"],
    "uzman-psikolog": ["uzm-psk", "uzm-kl-psk"],
    "doktor": ["dr"],
    "dis-hekimi": ["dt"],
    "fizyoterapist": ["fzt"],
    "veteriner": ["veteriner-hekim"],
    "ergoterapist": ["ergoterapist"],
    "cocuk-gelisim-uzmani": ["cocuk-gelisim-uzmani"],
    "dil-ve-konusma-terapisti": ["dil-ve-konusma-terapisti"],
    "pedagog": ["pedagog"]
}

class PreProcessor:
    def __init__(self, data):
        self.data = data

    def preprocess(self):
      df = self.data
      for key in speciality_mapping:
          df.loc[df['doctor_speciality'].isin(speciality_mapping[key]), 'doctor_speciality'] = key

      for key in specialty_mapping2:
          df.loc[df['doctor_speciality'].isin(specialty_mapping2[key]), 'doctor_speciality'] = key


      for key in title_mapping:
          df.loc[df['doctor_title'].isin(title_mapping[key]), 'doctor_title'] = key

      for key in title_mapping2:
          df.loc[df['doctor_title'].isin(title_mapping2[key]), 'doctor_title'] = key

      df['text'] = df['question_content'] + " " + df['question_answer']
      df = df.drop(['question_content', 'question_answer'], axis=1)
      df

      return df