from ultralytics import YOLO
import cv2
import math

modelo = YOLO(r"runs\pose\train-2\weights\best.pt")
cap = cv2.VideoCapture(0)

# --- CONFIGURAÇÕES ---
PERCENTAGEM_ZONA_CRITICA = 0.28
PERCENTAGEM_VELOCIDADE = 0.05
LIMITE_GOLPES = 4
TEMPO_TRAVA = 20  # Mantém o alerta no ecrã durante 20 frames para não piscar

contador_golpes = 0
tempo_restante_alerta = 0

# Variáveis de Memória (para suavizar os movimentos)
centro_cabeca_suav = None
pontos_anteriores = [(0,0), (0,0), (0,0)]

print("A iniciar a câmara... Pressione 'q' para sair.")

while cap.isOpened():
    sucesso, frame = cap.read()
    if not sucesso:
        break

    resultados = modelo(frame, conf=0.5)
    imagem_anotada = resultados[0].plot()
    
    raio_critico = 0
    centro_desenho = None

    if len(resultados[0]) > 0 and resultados[0].keypoints is not None and len(resultados[0].keypoints.xy) > 0:
        pontos = resultados[0].keypoints.xy[0]
        caixa = resultados[0].boxes.xyxy[0]
        
        altura_pessoa = float(caixa[3] - caixa[1])
        largura_pessoa = float(caixa[2] - caixa[0])
        
        # Posição crua do frame atual
        cabeca_x_atual = float(caixa[0]) + (largura_pessoa / 2)
        cabeca_y_atual = float(caixa[1]) + (altura_pessoa * 0.15) 
        
        # 1. FILTRO DE SUAVIZAÇÃO: Mistura 30% da posição nova com 70% da posição antiga
        if centro_cabeca_suav is None:
            centro_cabeca_suav = (cabeca_x_atual, cabeca_y_atual)
        else:
            novo_x = (cabeca_x_atual * 0.3) + (centro_cabeca_suav[0] * 0.7)
            novo_y = (cabeca_y_atual * 0.3) + (centro_cabeca_suav[1] * 0.7)
            centro_cabeca_suav = (novo_x, novo_y)

        centro_desenho = (int(centro_cabeca_suav[0]), int(centro_cabeca_suav[1]))
        raio_critico = int(altura_pessoa * PERCENTAGEM_ZONA_CRITICA)
        VELOCIDADE_MINIMA = altura_pessoa * PERCENTAGEM_VELOCIDADE
        
        if resultados[0].keypoints.conf is not None:
            confiancas = resultados[0].keypoints.conf[0]
            
            if len(pontos) >= 3 and len(confiancas) >= 3:
                houve_golpe = False
                
                for i in range(3):
                    pt_x, pt_y = float(pontos[i][0]), float(pontos[i][1])
                    conf = float(confiancas[i])
                    
                    if pt_x > 0 and pt_y > 0 and conf > 0.35:
                        dist_para_cabeca = math.hypot(centro_cabeca_suav[0] - pt_x, centro_cabeca_suav[1] - pt_y)
                        
                        pt_ant_x, pt_ant_y = pontos_anteriores[i]
                        velocidade = 0
                        if pt_ant_x > 0:
                            velocidade = math.hypot(pt_x - pt_ant_x, pt_y - pt_ant_y)
                        
                        pontos_anteriores[i] = (pt_x, pt_y)
                        
                        if dist_para_cabeca < raio_critico and velocidade > VELOCIDADE_MINIMA:
                            houve_golpe = True
                            
                if houve_golpe:
                    contador_golpes += 1
                else:
                    if contador_golpes > 0:
                        contador_golpes -= 0.5
                        
                # 2. SISTEMA DE TRAVA
                if contador_golpes >= LIMITE_GOLPES:
                    tempo_restante_alerta = TEMPO_TRAVA # Reinicia o cronómetro do aviso
                    contador_golpes = LIMITE_GOLPES

    if centro_desenho is not None and raio_critico > 0:
        cv2.circle(imagem_anotada, centro_desenho, raio_critico, (0, 255, 255), 2)

    # Exibe o alerta enquanto a trava estiver ativa
    if tempo_restante_alerta > 0:
        cv2.putText(imagem_anotada, "CRISE DETETADA!", (50, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 4)
        tempo_restante_alerta -= 1

    cv2.imshow("Teste - Detector de Crises", imagem_anotada)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()