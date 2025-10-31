import os
import subprocess

def run_all_snippets(folder_path: str):
    """
    Esegue tutti i file .py in una cartella e conta quanti terminano con successo
    e quanti falliscono.
    """

    # Verifica che la cartella esista
    if not os.path.isdir(folder_path):
        print(f"❌ Errore: la cartella '{folder_path}' non esiste.")
        return

    # Lista di file Python nella cartella (non ricorsiva)
    files = [f for f in os.listdir(folder_path) if f.endswith(".py")]
    if not files:
        print(f"Nessun file .py trovato in '{folder_path}'.")
        return

    success_count = 0
    fail_count = 0

    print(f"\n🚀 Avvio esecuzione di {len(files)} file Python nella cartella '{folder_path}'...\n")

    for filename in files:
        file_path = os.path.join(folder_path, filename)
        try:
            # Esegue il file e cattura output e errori
            result = subprocess.run(
                ["python", file_path],
                capture_output=True,
                text=True,
                timeout=10  # massimo 10 secondi per ogni script
            )

            if result.returncode == 0:
                print(f"✅ {filename} eseguito con successo.")
                success_count += 1
            else:
                print(f"❌ {filename} fallito con errore:\n{result.stderr.strip()}\n")
                fail_count += 1

        except subprocess.TimeoutExpired:
            print(f"⏰ {filename} terminato per timeout (più di 10 secondi).")
            fail_count += 1
        except Exception as e:
            print(f"⚠️ Errore imprevisto con {filename}: {e}")
            fail_count += 1

    # Report finale
    print("\n📊 Risultati finali:")
    print(f"   - File eseguiti con successo: {success_count}")
    print(f"   - File falliti: {fail_count}")
    print(f"   - Totale: {len(files)}")

if __name__ == "__main__":
    # Modifica qui il percorso della cartella che vuoi testare
    folder = (r"C:\Users\justl\PycharmProjects\thesis-green-ai-llm\risultati\refac_gpt5mini\Mixed")  # esempio: ./snippets
    run_all_snippets(folder)
