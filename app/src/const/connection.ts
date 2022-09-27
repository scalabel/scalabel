/* endpoint names for python server */
export const enum ModelEndpoint {
  PREDICT_POLY = "predictPoly",
  REFINE_POLY = "refinePoly"
}

/* endpoint names for http server */
export const enum Endpoint {
  POST_PROJECT = "/postProject",
  POST_PROJECT_INTERNAL = "/postProjectInternal",
  GET_PROJECT_NAMES = "/getProjectNames",
  EXPORT = "/getExport",
  DASHBOARD = "/getDashboardContents",
  GET_TASK_METADATA = "/getTaskMetaData",
  POST_TASKS = "/postTasks",
  CALLBACK = "/callback",
  STATS = "/stats",
  DELETE_PROJECT = "/deleteProject"
}

/* socket.io event names */
export const enum EventName {
  ACTION_BROADCAST = "actionBroadcast",
  ACTION_SEND = "actionSend",
  REGISTER_ACK = "registerAck",
  REGISTER = "register",
  CONNECTION = "connection",
  CONNECT = "connect",
  DISCONNECT = "disconnect"
}

/* redis channels */
export const enum RedisChannel {
  REGISTER_EVENT = "registerEvent",
  // CONNECTION = "CONNECTIONS",
  // CONNECTION_RESPONSE = "SCALABEL_CONNECTION_RESPONSE",
  MODEL_REGISTER = "modelRegister",
  MODEL_REQUEST = "taskRequests",
  MODEL_RESPONSE = "responses",
  MODEL_STATUS = "modelStatus",
  MODEL_NOTIFY = "modelNotify"
}
