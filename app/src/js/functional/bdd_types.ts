import {
  ConfigType, CubeType, ItemType,
  PolygonType, RectType, TaskStatus, TrackType } from './types'

export interface TaskData {
  /** task config data that's constant throughout a session */
  config: ConfigType,
  /** current status of task */
  status: TaskStatus,
  /** items in task */
  Items: ItemType[],
  /** tracks for task */
  Tracks: TrackType
}

export interface ItemExport {
  /** project name */
  name: string
  /** item url */
  url: string
  /** item videoname */
  videoName: string
  /** item attributes */
  attributes: { [key: string]: string[] }
  /** submitted timestamp */
  timestamp: number
  /** item index */
  index: number
  /** item labels */
  labels: LabelExport[]
}

export interface LabelExport {
  /** label id */
  id: number
  /** category */
  category: string
  /** label attributes */
  attributes: { [key: string]: string[] }
  /** if shape was manual */
  manualShape: boolean
  /** box2d label */
  box2d: RectType | null
  /** poly2d label */
  poly2d: PolygonType | null
  /** box3d label */
  box3d: CubeType | null
}
