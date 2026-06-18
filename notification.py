import requests

def notificar_ntfy(mensagem):
    # Substitua pelo nome exato do tópico que você criou no app
    topico = "meu_modelo_secreto_xyz_2026" 
    url = f"https://ntfy.sh/{topico}"
    
    # Opcional: Adiciona um título à notificação
    headers = {
        "Title": "Python Notifier",
        "Tags": "tada" # Adiciona um emoji
    }
    
    try:
        requests.post(url, data=mensagem.encode('utf-8'), headers=headers)
        print("Notificação enviada com sucesso!")
    except Exception as e:
        print(f"Erro ao enviar notificação: {e}")


# notificar_ntfy("Seu modelo acabou de rodar")
