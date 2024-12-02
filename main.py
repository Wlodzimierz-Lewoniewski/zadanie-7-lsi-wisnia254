import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


def zbuduj_macierz(dokumenty, zapytanie):
    wszystkie_slowa = set()
    for dok in dokumenty + [zapytanie]:
        wszystkie_slowa.update(dok.lower().replace(".", "").split())
    wszystkie_slowa = sorted(wszystkie_slowa)

    macierz = np.zeros((len(wszystkie_slowa), len(dokumenty)))
    for j, dok in enumerate(dokumenty):
        for slowo in dok.lower().replace(".", "").split():
            macierz[wszystkie_slowa.index(slowo), j] = 1
    return macierz, wszystkie_slowa


liczba_dok = int(input())
dokumenty = [input().strip() for _ in range(liczba_dok)]
zapytanie = input().strip()
wymiary = int(input())

macierz_dok, slowa = zbuduj_macierz(dokumenty, zapytanie)

U, S, Vt = np.linalg.svd(macierz_dok, full_matrices=False)

U_przyciete = U[:, :wymiary]
S_przyciete = np.diag(S[:wymiary])
Vt_przyciete = Vt[:wymiary, :]

dokumenty_zredukowane = S_przyciete @ Vt_przyciete

wektor_zapytania = np.zeros(len(slowa))
for slowo in zapytanie.lower().split():
    if slowo in slowa:
        wektor_zapytania[slowa.index(slowo)] = 1

zapytanie_zredukowane = np.linalg.inv(S_przyciete) @ (U_przyciete.T @ wektor_zapytania)

podobienstwa = cosine_similarity(zapytanie_zredukowane.reshape(1, -1), dokumenty_zredukowane.T).flatten()

wyniki = np.round(podobienstwa, 2)

wyniki_list = list(map(float, wyniki))

print(wyniki_list)
