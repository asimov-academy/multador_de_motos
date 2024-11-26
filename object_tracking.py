from deep_sort_realtime.deepsort_tracker import DeepSort
from config import DEEPSORT_MAX_AGE

def inicializar_rastreadores():
    """Inicializa os rastreadores para motocicletas e semáforos."""
    tracker_motos = DeepSort(max_age=DEEPSORT_MAX_AGE)
    tracker_semaforos = DeepSort(max_age=DEEPSORT_MAX_AGE)
    return tracker_motos, tracker_semaforos

def atualizar_rastreadores(tracker_motos, tracker_semaforos, det_motos, det_semaforos, frame):
    """Atualiza os rastreadores com as detecções atuais."""
    tracks_motos = tracker_motos.update_tracks(det_motos, frame=frame)
    tracks_semaforos = tracker_semaforos.update_tracks(det_semaforos, frame=frame)
    return tracks_motos, tracks_semaforos
