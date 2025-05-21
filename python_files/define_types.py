import pandas as pd

clean_data_cut = pd.read_csv('D:/ProjectsVSCode/classificator_stars/csv/data_with_types2.csv')

def classify_type(vtype: str) -> str:
    """
    Возвращает укрупнённый класс переменной звезды
    в зависимости от содержимого строки vtype.
    """

    if pd.isna(vtype) or not isinstance(vtype, str) or vtype.strip() == "":
        return "UNKNOWN"  # Или "UNKNOWN", если хотите отдельно отметить NaN и пустые

    # Приведём к верхнему регистру для надёжного поиска подстрок
    t = vtype.upper()

    # --- 1) Затменные (Eclipsing Binaries) ---
    # Ищем любые подтипы: EA, EB, EW, EC, ELL, E/RS и т.д.
    # Включим также "E|" (встречается в комбинированных типах) и просто "E" (бывает)
    ecl_markers = ["EA", "EB", "EW", "EC", "ELL", "E/RS", "E|", "E "]
    if any(m in t for m in ecl_markers):
        return "ECLIPSING"

    # --- 2) Цефеиды и родственные (DCEP, CW, RV Tauri, ACEP) ---
    cep_markers = ["DCEP", "CW-FU", "CW", "CWA", "CWB", "RVA", "RV", "ACEP", "CEP"]
    # Примечание: "RV" может пересекаться с "RVC", "ROT" и пр.,
    # поэтому в реальном коде можно уточнять условия, но здесь — упрощённо.
    if any(m in t for m in cep_markers):
        return "CEPHEIDS"

    # --- 3) RR Лиры (RRAB, RRC, RRD, RR...) ---
    rr_markers = ["RRAB", "RRC", "RRD", "RR"]
    if any(m in t for m in rr_markers):
        return "RR_LYRAE"

    # --- 4) Короткопериодические пульсаторы: DSCT, SXPHE, GDOR, roAp ---
    short_puls = ["DSCT", "HADS", "SXPHE", "GDOR", "ROAP", "ROAM"]
    if any(m in t for m in short_puls):
        return "DELTA_SCUTI_ETC"

    # --- 5) Долгопериодические и полуправильные (M, SR, L) ---
    # Mira (M), SR, SRA, SRB, SRC, SRD, L, LB, LC, LPV
    lpv_markers = [" M ", "MIRA", "SR", "SRA", "SRB", "SRC", "SRD", "L ", "LB", "LC", "LPV"]
    # Для "M" можно проверять отдельно, чтобы не совпадало с "MISC", поэтому " M " с пробелами,
    # но тут для упрощения — любой "M". В реальном коде нужна аккуратность или RegEx.
    if any(m in t for m in lpv_markers):
        return "LONG_PERIOD"

    # --- 6) Ротационные переменные (BY, RS, ACV, SPB, ROT, GCAS) ---
    rot_markers = ["BY", "RS", "ACV", "SPB", "ROT", "GCAS"]
    if any(m in t for m in rot_markers):
        return "ROTATIONAL"

    # --- 7) Эруптивные/молодые звёзды (T Tauri, EXOR, UXOR, INS...) ---
    yso_markers = ["TTS", "EXOR", "UXOR", "INS", "IN", "INST", "CST"]  
    # "CST" иногда "constant?", но бывает и у молодых/неясных
    if any(m in t for m in yso_markers):
        return "YSO/ERUPTIVE"

    # --- Катаклизмические (UG, NL, AM, ZAND, IB, IS, ... ) ---
    cataclysmic_markers = ["UG", "NL", "AM", "ZAND", "IB", "ISB"]  # и др.
    if any(m in t for m in cataclysmic_markers):
        return "CATACLYSMIC"

    # --- 9) Горячие эмиссионные/WR/Be/симбиотические (WR, BE, FSCMa...) ---
    em_markers = ["WR", "BE", "FSCMA"]
    if any(m in t for m in em_markers):
        return "EMISSION_WR"

    # Если ничего не подошло, отправляем в "MISC" или "HYBRID"
    return "UNKNOWN"


# Теперь создаём новую колонку:
clean_data_cut["class"] = clean_data_cut["type"].apply(classify_type)

print(clean_data_cut[clean_data_cut["class"] != 'UNKNOWN'])

# убрать "YSO/ERUPTIVE" и "EMISSION_WR"
# clean_data_cut = clean_data_cut[clean_data_cut['class'] != 'YSO/ERUPTIVE']
# clean_data_cut = clean_data_cut[clean_data_cut['class'] != 'EMISSION_WR']
# clean_data_cut = clean_data_cut[clean_data_cut['class'] != 'UNKNOWN']
# clean_data_cut = clean_data_cut[clean_data_cut['class'] != 'CEPHEIDS']
# clean_data_cut = clean_data_cut[clean_data_cut['class'] != 'CATACLYSMIC']

# clean_data_cut.to_csv('D:/ProjectsVSCode/classificator_stars/csv/types_defined.csv', index=False, sep=',')