import { ModelEndpoint } from "../const/connection"
import { BaseAction } from "./action"
import { ItemExport } from "./export"

export interface RegisterMessageType {
  /** Project name of the session */
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
