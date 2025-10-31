arr = [1, 2, 3, 4, 5, 6]
import random
random.seed(13)  # Imposta il seme per la riproducibilità
for i in range(len(arr) - 1, 0, -1):
    j = random.randint(0, i)  # Usa una funzione built-in per la generazione di numeri casuali
    arr[i], arr[j] = arr[j], arr[i]
print(arr)