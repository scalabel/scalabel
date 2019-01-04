from concurrent import futures
import datetime
import grpc
import model_server_pb2 as pb2
import model_server_pb2_grpc as pb2_grpc
import time
import ray
import logging
logging.basicConfig(level=logging.INFO)


@ray.remote
class SessionWorker():
    def __init__(self, sessionId):
        self.sessionId = sessionId

    def do_work(self):
        return str(datetime.datetime.now())


class ModelServer(pb2_grpc.ModelServerServicer):
    def __init__(self):
        super().__init__()
        self.sessionIdsToWorkers = {}

    def Register(self, request, context):
        start = time.time()
        if request.sessionId not in self.sessionIdsToWorkers:
            newWorker = SessionWorker.remote(request.sessionId)
            self.sessionIdsToWorkers[request.sessionId] = newWorker
        timestamp = str(datetime.datetime.now())
        end = time.time()
        duration = "{0:.3f}".format((end - start) * 1000.0)
        return pb2.Response(session=request,
                            modelServerTimestamp=timestamp,
                            modelServerDuration=duration)

    def DummyComputation(self, request, context):
        start = time.time()
        id = request.sessionId
        worker = self.sessionIdsToWorkers[id]
        timestamp = ray.get(worker.do_work.remote())
        logging.info(f'Got this message {request} at {timestamp}')
        end = time.time()
        duration = "{0:.3f}".format((end - start) * 1000.0)
        return pb2.Response(session=request,
                            modelServerTimestamp=timestamp,
                            modelServerDuration=duration)

    def KillActor(self, request, context):
        id = request.sessionId
        worker = self.sessionIdsToWorkers.pop(id, None)
        if worker:
            del worker
            logging.info(f'deleted worker for id {id}')
        else:
            loggin.info(f'attempted to delete worker for id {id}'
                        + ' but none exists')
        return pb2.Empty()


def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=100))
    pb2_grpc.add_ModelServerServicer_to_server(ModelServer(), server)
    server.add_insecure_port('[::]:50051')
    server.start()
    try:
        while True:
            time.sleep(300000)
    except KeyboardInterrupt:
        server.stop(0)


if __name__ == '__main__':
    ray.init(num_cpus=100, ignore_reinit_error=True)
    serve()
