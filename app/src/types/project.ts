import { ItemExport } from "./export"
import { Attribute, ConfigType, Label2DTemplateType, SensorType } from "./state"

/**
 * Stores specifications of project
 */
export interface Project {
  /** array of items */
  items: Array<Partial<ItemExport>>
  /** frontend config */
  config: ConfigType
  /** map between data source id and data sources */
  sensors: { [id: number]: SensorType }
}

/**
 * Form data for project creation
 */
export interface CreationForm {
  /** name of project */
  projectName: string
  /** item type */
  itemType: string
  /** label type */
  labelType: string
  /** title of page */
  pageTitle: string
  /** task size */
  taskSize: number
  /** instruction url */
  instructionUrl: string
  /** whether demo mode is true */
  demoMode: boolean
}

/* file data parsed from form */
export interface FormFileData {
  /** categories parsed from form file */
  categories: string[]
  /** sensors */
  sensors: SensorType[]
  /** custom label template */
  templates: Label2DTemplateType[]
  /** attributes parsed from form file */
  attributes: Attribute[]
  /** items parsed from form file (may be incomplete) */
  items: Array<Partial<ItemExport>>
}

/** metadata associated with a state */
// TODO: rename this interface from Metadata to task info
export interface StateMetadata {
  /** project name */
  projectName: string
  /** task id */
  taskId: string
  /** map from processed action ids to their timestamps */
  actionIds: { [key: string]: number[] }
}

/** user data for a project */
export interface UserData {
  /** project name */
  projectName: string
  /** map from socket to user */
  socketToUser: { [key: string]: string }
  /** map from user to list of socket */
  userToSockets: { [key: string]: string[] }
}

/** metadata for all users for all projects */
export interface UserMetadata {
  /** map from socket to project */
  socketToProject: { [key: string]: string }
}
