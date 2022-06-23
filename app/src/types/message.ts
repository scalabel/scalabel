import { ModelEndpoint } from "../const/connection"
import { BaseAction } from "./action"
import { ItemType } from "./state"
import { ItemExport } from "./export"
import { ModelRequestType } from "../const/common"

export interface RegisterMessageType {
  /** Project name */
  projectName: string
  /** Task index of the session */
  taskIndex: number
  /** Current session Id */
  sessionId: string
  /** Current user Id */
  userId: string
  /** server address */
  address: string
  /** whether it came from a bot or not */
  bot: boolean
  /** label type */
  labelType?: string
}

/** action type for synchronization between front and back ends */
export interface SyncActionMessageType {
  /** Task Id. It is supposed to be index2str(taskIndex) */
  taskId: string
  /** Project name */
  projectName: string
  /** Session Id */
  sessionId: string
  /** List of actions for synchronization */
  actions: ActionPacketType
  /** whether it came from a bot or not */
  bot: boolean
}

/** model register message type */
export interface ModelRegisterMessageType {
  /** Project name */
  projectName: string
  /** Task Id. It is supposed to be index2str(taskIndex) */
  taskId: string
  /** item list */
  items: ItemType[]
  /** task type */
  taskType: string
}

/** model register message type */
export interface ModelStatusMessageType {
  /** Project name */
  projectName: string
  /** Task Id. It is supposed to be index2str(taskIndex) */
  taskId: string
  /** if true, set to active, else to inactive */
  active: boolean
}

/** model request message type */
export interface ModelRequestMessageType {
  /** Request type */
  type: ModelRequestType
  /** Task type */
  taskType: string
  /** Project name */
  projectName: string
  /** Task Id. It is supposed to be index2str(taskIndex) */
  taskId: string
  /** item list */
  items: ItemExport[]
  /** item indices list*/
  itemIndices: number[]
  /** which action triggers this prediction request */
  actionPacketId: string
}

/** type for transmitted packet of actions */
export interface ActionPacketType {
  /** list of actions in the packet */
  actions: BaseAction[]
  /** id of the packet */
  id: string
  /** for bot actions, id of the action packet that triggered them */
  triggerId?: string
}

/** data kept by each bot user */
export interface BotData {
  /** the project name */
  projectName: string
  /** the index of the task */
  taskIndex: number
  /** the bot user id */
  botId: string
  /** the address of the io server */
  address: string
  /** label type */
  labelType: string
}

/** precomputed queries for models */
export interface ModelQuery {
  /** the data in scalabel format */
  data: ItemExport
  /** the endpoint for the query */
  endpoint: ModelEndpoint
  /** the index of the item modified */
  itemIndex: number
}

/** the form of request sent to bot session */
export interface ModelRequest {
  /** the data in scalabel format */
  data: ItemExport[]
  /** the index of the item modified */
  itemIndices: number[]
}
