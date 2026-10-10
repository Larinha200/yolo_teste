from ultralytics import YOLO

def iniciar_treino():
    # Carrega o modelo de Pose inicial (esqueleto)
    model = YOLO("yolov8n-pose.pt")

    print("A iniciar o treino... Isto pode demorar alguns minutos ou horas!")
    
    # Inicia o treino
    # Repare que alterámos o caminho do data para encontrar o ficheiro solto na pasta principal
    results = model.train(
        data="data.yaml", 
        epochs=50, 
        imgsz=640,
        plots=True 
    )
    
    print("Treino concluído com sucesso!")

if __name__ == '__main__':
    iniciar_treino()