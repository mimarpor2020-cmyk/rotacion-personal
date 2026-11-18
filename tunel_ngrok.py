"""Publica la aplicación con un enlace público usando ngrok (pyngrok).

Uso:
    1. Crea una cuenta gratis en https://dashboard.ngrok.com/signup y copia tu authtoken.
    2. Windows (PowerShell):  $env:NGROK_AUTHTOKEN="tu_token"
       Linux / macOS:          export NGROK_AUTHTOKEN=tu_token
       (si no la defines, el script te la pedirá)
       Para las recomendaciones del LLM define también GROQ_API_KEY (https://console.groq.com/keys).
    3. python tunel_ngrok.py
    4. Comparte el enlace https://....ngrok-free.app que aparece en consola.
"""
import getpass
import os
import sys

from pyngrok import conf, ngrok

PUERTO = int(os.environ.get("PUERTO", 5000))


def main():
    token = os.environ.get("NGROK_AUTHTOKEN") or getpass.getpass("Pega tu NGROK_AUTHTOKEN: ").strip()
    if not token:
        sys.exit("Se necesita un authtoken de ngrok (https://dashboard.ngrok.com/get-started/your-authtoken).")
    conf.get_default().auth_token = token

    # Clave del LLM (Groq). Sin ella la app funciona, pero sin recomendaciones.
    if not os.environ.get("GROQ_API_KEY"):
        try:
            clave = getpass.getpass("Pega tu GROQ_API_KEY (Enter para omitir): ").strip()
        except EOFError:
            clave = ""
        if clave:
            os.environ["GROQ_API_KEY"] = clave

    # Se importa la app aquí para cargar el modelo antes de abrir el túnel
    from app import app, ERROR_CARGA
    if ERROR_CARGA:
        print("ATENCIÓN:", ERROR_CARGA)

    tunel = ngrok.connect(PUERTO, "http")
    print("\n" + "=" * 64)
    print(" Aplicación publicada en:", tunel.public_url)
    print(" Local:                    http://127.0.0.1:%d" % PUERTO)
    print(" Ctrl + C para detener el servidor y cerrar el túnel")
    print("=" * 64 + "\n")
    try:
        app.run(host="0.0.0.0", port=PUERTO, debug=False, use_reloader=False)
    finally:
        ngrok.kill()


if __name__ == "__main__":
    main()
