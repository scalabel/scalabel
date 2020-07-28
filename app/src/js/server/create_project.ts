import * as fs from 'fs-extra'
import * as yaml from 'js-yaml'
import _ from 'lodash'
import { getInstructionUrl, getPageTitle, getTracking, index2str } from '../common/util'
import { ItemTypeName, LabelTypeName } from '../const/common'
import { FormField } from '../const/project'
import { isValidId, makeSensor, makeTask, makeTrack } from '../functional/states'
import { ItemExport } from '../types/bdd'
import { MaybeError } from '../types/common'
import {
  Attribute,
  ConfigType,
  ItemType,
  Label2DTemplateType,
  SensorType,
  TaskStatus,
  TaskType,
  TrackIdMap
} from '../types/functional'
import { CreationForm, FormFileData, Project } from '../types/project'
import * as defaults from './defaults'
import { convertItemToImport } from './import'
import { ProjectStore } from './project_store'
import * as util from './util'

/**
 * convert fields to form and validate input
 * if invalid input is found, error is returned to user via alert
 */
export async function parseForm (
  fields: { [key: string]: string},
  projectStore: ProjectStore): Promise<CreationForm> {
  // Check that required fields were entered
  let projectName = fields[FormField.PROJECT_NAME]
  if (projectName === '') {
    throw(Error('Please create a project name'))
  } else {
    projectName = util.parseProjectName(projectName)
  }

  const itemType = fields[FormField.ITEM_TYPE]
  if (itemType === '') {
    throw(Error('Please choose an item type'))
  }

  const labelType = fields[FormField.LABEL_TYPE]
  if (labelType === '') {
    throw(Error('Please choose a label type'))
  }

  // Task size is not required for videos
  let taskSize = 1 // Video case
  if (fields[FormField.ITEM_TYPE] !== ItemTypeName.VIDEO) {
    if (fields[FormField.TASK_SIZE] === '') {
      throw(Error('Please specify a task size'))
    } else {
      taskSize = parseInt(fields[FormField.TASK_SIZE], 10)
    }
  }

  // Derived fields
  const pageTitle = getPageTitle(labelType, itemType)
  const instructionUrl = getInstructionUrl(labelType)

  // Ensure project name is not already in use
  const exists = await projectStore.checkProjectName(projectName)
  if (exists) {
    throw(Error('Project name already exists.'))
  }
  const demoMode = fields[FormField.DEMO_MODE] === 'true'
  const form = util.makeCreationForm(
    projectName, itemType, labelType, pageTitle, taskSize,
    instructionUrl, demoMode
  )
  return form
}

/**
 * Parses item, category, and attribute files from paths
 */
export async function parseFiles (
  labelType: string, files: { [key: string]: string }, itemsRequired: boolean)
  : Promise<FormFileData> {
  return Promise.all([
    parseItems(files, itemsRequired),
    parseSensors(files),
    parseTemplates(files),
    parseAttributes(files, labelType),
    parseCategories(files, labelType)])
    .then((result: [
      Array<Partial<ItemExport>>,
      SensorType[],
      Label2DTemplateType[],
      Attribute[],
      string[]
    ]) => {
      return {
        items: result[0],
        sensors: result[1],
        templates: result[2],
        attributes: result[3],
        categories: result[4]
      }
    })
}

/**
 * Get default attributes if they weren't provided
 */
function getDefaultCategories (labelType: string): string[] {
  switch (labelType) {
    // TODO: add seg2d defaults (requires subcategories)
    case LabelTypeName.BOX_3D:
    case LabelTypeName.BOX_2D:
      return defaults.boxCategories
    case LabelTypeName.POLYLINE_2D:
      return defaults.polyline2DCategories
    default:
      return []
  }
}

/**
 * Read categories from yaml file at path
 */
function readCategoriesFile (path: string): Promise<string[]> {
  return new Promise((resolve, reject) => {
    fs.readFile(path, 'utf8', (err: MaybeError, file: string) => {
      if (err) {
        reject(err)
        return
      }
      // TODO: support subcategories
      const categories = yaml.load(file)
      const categoriesList = []
      for (const category of categories) {
        categoriesList.push(category.name)
      }
      resolve(categoriesList)
    })
  })
}

/**
 * Load from category file
 * Use default if file is empty
 */
export function parseCategories (
  files: { [key: string]: string },
  labelType: string): Promise<string[]> {
  if (FormField.CATEGORIES in files) {
    return readCategoriesFile(files[FormField.CATEGORIES])
  } else {
    const categories = getDefaultCategories(labelType)
    return Promise.resolve(categories)
  }
}

/**
 * Get default attributes if they weren't provided
 */
function getDefaultAttributes (labelType: string): Attribute[] {
  switch (labelType) {
    case LabelTypeName.BOX_2D:
      return defaults.box2DAttributes
    default:
      return defaults.dummyAttributes
  }
}

/**
 * Read attributes from yaml file at path
 */
function readAttributesFile (path: string): Promise<Attribute[]> {
  return new Promise((resolve, reject) => {
    fs.readFile(path, 'utf8', (err: MaybeError, fileBytes: string) => {
      if (err) {
        reject(err)
        return
      }
      const attributes = yaml.load(fileBytes)
      resolve(attributes)
    })
  })
}

/**
 * Load from attributes file
 * Use default if file is empty
 */
export function parseAttributes (
  files: { [key: string]: string },
  labelType: string): Promise<Attribute[]> {
  if (FormField.ATTRIBUTES in files) {
    return readAttributesFile(files[FormField.ATTRIBUTES])
  } else {
    const defaultAttributes = getDefaultAttributes(labelType)
    return Promise.resolve(defaultAttributes)
  }
}

/**
 * Read items from yaml file at path
 */
export function readItemsFile (
  path: string): Promise<Array<Partial<ItemExport>>> {
  return new Promise((resolve, reject) => {
    fs.readFile(path, 'utf8', (err: MaybeError, fileBytes: string) => {
      if (err) {
        reject(err)
        return
      }
      try {
        // Might not have all fields defined, so use partial
        const items =
          yaml.safeLoad(fileBytes, { json: true }) as Array<Partial<ItemExport>>
        resolve(items)
      } catch {
        reject(Error('Improper formatting for items file'))
      }
    })
  })
}

/**
 * Load from items file
 * Group by video name
 */
export function parseItems (
  files: { [key: string]: string },
  itemsRequired: boolean): Promise<Array<Partial<ItemExport>>> {
  if (FormField.ITEMS in files) {
    return readItemsFile(files[FormField.ITEMS])
  } else {
    if (itemsRequired) {
      return Promise.reject(Error('No item file.'))
    } else {
      return Promise.resolve([])
    }
  }
}

/** Read sensors file */
function readSensorsFile (path: string): Promise<SensorType[]> {
  return new Promise((resolve, reject) => {
    fs.readFile(path, 'utf8', (err: MaybeError, fileBytes: string) => {
      if (err) {
        reject(err)
      } else {
        try {
          const sensors = yaml.load(fileBytes) as SensorType[]
          resolve(sensors)
        } catch {
          reject(Error('Improper formatting for sensors file'))
        }
      }
    })
  })
}

/** Parse files for sensors */
export function parseSensors (
  files: { [key: string]: string }): Promise<SensorType[]> {
  if (FormField.SENSORS in files) {
    return readSensorsFile(files[FormField.SENSORS])
  } else {
    return Promise.resolve([])
  }
}

/** Read sensors file */
function readTemplatesFile (path: string): Promise<Label2DTemplateType[]> {
  return new Promise((resolve, reject) => {
    fs.readFile(path, 'utf8', (err: MaybeError, fileBytes: string) => {
      if (err) {
        reject(err)
      } else {
        try {
          const templates = yaml.load(fileBytes) as Label2DTemplateType[]
          resolve(templates)
        } catch {
          reject(Error('Improper formatting for sensors file'))
        }
      }
    })
  })
}

/** Parse files for sensors */
export function parseTemplates (
  files: { [key: string]: string }): Promise<Label2DTemplateType[]> {
  if (files[FormField.LABEL_SPEC] in files) {
    return readTemplatesFile(files[FormField.LABEL_SPEC])
  } else {
    return Promise.resolve([])
  }
}

/**
 * Marshal data into project format
 */
export function createProject (
  form: CreationForm,
  formFileData: FormFileData): Promise<Project> {

  const handlerUrl = util.getHandlerUrl(form.itemType, form.labelType)
  const bundleFile = util.getBundleFile(form.labelType)
  const [itemType, tracking] = getTracking(form.itemType)

  const templates: { [name: string]: Label2DTemplateType } = {}

  for (let i = 0; i < formFileData.templates.length; i++) {
    templates[`template${i}`] = formFileData.templates[i]
  }

  /* use arbitrary values for
   * taskId and policyTypes
   * assign these when tasks are created
   */
  const config: ConfigType = {
    projectName: form.projectName,
    itemType,
    labelTypes: [form.labelType],
    label2DTemplates: templates,
    taskSize: form.taskSize,
    handlerUrl,
    pageTitle: form.pageTitle,
    instructionPage: form.instructionUrl,
    bundleFile,
    categories: formFileData.categories,
    attributes: formFileData.attributes,
    taskId: '',
    tracking,
    policyTypes: [],
    demoMode: form.demoMode,
    autosave: true,
    bots: false
  }

  // Ensure that all video names are set to default if empty
  let projectItems = formFileData.items
  projectItems.forEach((itemExport) => {
    if (itemExport.videoName === undefined) {
      itemExport.videoName = ''
    }
  })

  const sensors: {[id: number]: SensorType} = {}
  for (const sensor of formFileData.sensors) {
    sensors[sensor.id] = sensor
  }

  // With tracking, order by video name lexicographically and split according
  // to video name. It should be noted that a stable sort must be used to
  // maintain ordering provided in the image list file
  projectItems = _.sortBy(projectItems, [(item) => item.videoName])
  const project: Project = {
    config,
    items: projectItems,
    sensors
  }
  return Promise.resolve(project)
}

/**
 * Create two maps for quick lookup of attribute data
 * @param configAttributes the attributes from config file
 * first RV: map from attribute name to attribute and its index
 * second RV: map from attribute value to its index within that attribute
 */
function getAttributeMaps (
  configAttributes: Attribute[]):
  [{[key: string]: [number, Attribute]}, {[key: string]: number}] {
  const attributeNameMap: {[key: string]: [number, Attribute]} = {}
  const attributeValueMap: {[key: string]: number} = {}
  for (let attrInd = 0; attrInd < configAttributes.length; attrInd++) {
    const configAttribute = configAttributes[attrInd]
    // Map attribute name to its index and its value
    attributeNameMap[configAttribute.name] = [attrInd, configAttribute]
    // Map attribute values to their indices (if its a list)
    if (configAttribute.toolType === 'list') {
      const values = configAttribute.values
      for (let valueInd = 0; valueInd < values.length; valueInd++) {
        const value = values[valueInd]
        attributeValueMap[value] = valueInd
      }
    }
  }
  return [attributeNameMap, attributeValueMap]
}

/**
 * Create a map for quick lookup of category data
 * @param configCategories the categories from config file
 * returns a map from category value to its index
 */
function getCategoryMap (
  configCategories: string[]): {[key: string]: number} {
  const categoryNameMap: {[key: string]: number} = {}
  for (let catInd = 0; catInd < configCategories.length; catInd++) {
    // Map category names to their indices
    const category = configCategories[catInd]
    categoryNameMap[category] = catInd
  }
  return categoryNameMap
}

/**
 * Filter invalid items, condition depends on whether labeling fusion data
 * Items are in export format
 */
function filterInvalidItems (
  items: Array<Partial<ItemExport>>, itemType: string,
  sensors: { [id: number]: SensorType}): Array<Partial<ItemExport>> {
  if (itemType === ItemTypeName.FUSION) {
    return items.filter((itemExport) =>
      itemExport.dataType === undefined &&
      itemExport.sensor !== undefined &&
      itemExport.timestamp !== undefined &&
      itemExport.sensor in sensors
    )
  } else {
    return items.filter((itemExport) =>
      !itemExport.dataType ||
      itemExport.dataType === itemType
    )
  }
}

/**
 * Partitions the item into tasks
 * Returns list of task indices in format [start, stop) for every task
 */
function partitionItemsIntoTasks (
  items: Array<Partial<ItemExport>>, tracking: boolean,
  taskSize: number): number[] {
  const taskIndices: number[] = []
  if (tracking) {
    // Partition by video name
    let prevVideoName: string
    items.forEach((value, index) => {
      if (value.videoName !== undefined) {
        if (value.videoName !== prevVideoName) {
          taskIndices.push(index)
          prevVideoName = value.videoName
        }
      }
    })
  } else {
    // Partition uniformly
    for (let i = 0; i < items.length; i += taskSize) {
      taskIndices.push(i)
    }
  }
  taskIndices.push(items.length)
  return taskIndices
}

/**
 * Map from data source id to list of items
 */
function mapSensorToItems (
  items: Array<Partial<ItemExport>>
): {[id: number]: Array<Partial<ItemExport>>} {
  const itemsBySensor: {[id: number]: Array<Partial<ItemExport>>} = {}
  for (const item of items) {
    const sensorId = item.sensor
    if (sensorId !== undefined) {
      if (!(sensorId in itemsBySensor)) {
        itemsBySensor[sensorId] = []
      }
      itemsBySensor[sensorId].push(item)
    }
  }
  return itemsBySensor
}

/**
 * Split project into tasks
 * Each consists of the task portion of a front  end state
 * Task and item start number are used if other tasks/items already exist
 */
export function createTasks (
  project: Project,
  taskStartNum: number = 0,
  itemStartNum: number = 0): Promise<TaskType[]> {
  const sensors = project.sensors
  const itemType = project.config.itemType
  const taskSize = project.config.taskSize
  const tracking = project.config.tracking

  const items = filterInvalidItems(
    project.items, itemType, sensors)

  // Update sensor info
  if (itemType !== ItemTypeName.FUSION) {
    sensors[-1] = makeSensor(-1, 'default', itemType)
    let maxSensorId =
      Math.max(...Object.keys(sensors).map((key) => Number(key)))
    for (const itemExport of items) {
      if (itemExport.dataType) {
        sensors[maxSensorId + 1] = makeSensor(
          maxSensorId,
          '',
          itemExport.dataType,
          itemExport.intrinsics,
          itemExport.extrinsics
        )
        itemExport.sensor = maxSensorId
        itemExport.dataType = undefined
        maxSensorId++
      } else if (
        itemExport.sensor === undefined ||
        !(itemExport.sensor in project.sensors)
      ) {
        itemExport.sensor = -1
      }
    }
  }

  const itemIndices = partitionItemsIntoTasks(
    items, tracking, taskSize)

  /* create quick lookup dicts for conversion from export type
   * to external type for attributes/categories
   * this avoids lots of indexof calls which slows down creation */
  const [attributeNameMap, attributeValueMap] = getAttributeMaps(
    project.config.attributes)
  const categoryNameMap = getCategoryMap(project.config.categories)
  const tasks: TaskType[] = []

  for (let taskIndex = 0; taskIndex < itemIndices.length - 1; taskIndex++) {
    const itemStartIndex = itemIndices[taskIndex]
    const itemEndIndex = itemIndices[taskIndex + 1]
    const taskItems = items.slice(itemStartIndex, itemEndIndex)
    const itemsBySensor = mapSensorToItems(taskItems)
    const sensorIds = Object.keys(itemsBySensor).map(Number)

    let realTaskSize = 0
    let largestSensor = -1
    const sensorMatchingIndices: {[id: number]: number} = {}
    for (const sensorId of sensorIds) {
      itemsBySensor[sensorId] = _.sortBy(
        itemsBySensor[sensorId],
        [(itemExport) => util.getItemTimestamp(itemExport)]
      )
      realTaskSize = Math.max(
        realTaskSize, itemsBySensor[sensorId].length
      )
      if (realTaskSize === itemsBySensor[sensorId].length) {
        largestSensor = sensorId
      }
      sensorMatchingIndices[sensorId] = 0
    }

    /* assign task id,
     and update task size in case there aren't enough items */
    const config: ConfigType = {
      ...project.config,
      taskSize: realTaskSize,
      taskId: index2str(taskStartNum + taskIndex)
    }

    // Based on the imported labels, compute max ids
    // max order is the total number of labels
    let maxOrder = 0

    // Convert from export format to internal format
    const itemsForTask: ItemType[] = []
    const trackMap: TrackIdMap = {}
    for (let itemInd = 0; itemInd < realTaskSize; itemInd += 1) {
      const timestampToMatch = itemsBySensor[largestSensor][
        sensorMatchingIndices[largestSensor]
      ].timestamp as number
      const itemExportMap: {[id: number]: Partial<ItemExport>} = {}
      for (const key of Object.keys(sensorMatchingIndices)) {
        const sensorId = Number(key)
        let newIndex = sensorMatchingIndices[sensorId]
        const itemExports = itemsBySensor[sensorId]
        while (newIndex < itemExports.length - 1 &&
               Math.abs(itemExports[newIndex + 1].timestamp as number -
                        timestampToMatch) <
               Math.abs(itemExports[newIndex].timestamp as number -
                        timestampToMatch)
        ) {
          newIndex++
        }
        sensorMatchingIndices[sensorId] = newIndex
        itemExportMap[sensorId] = itemExports[newIndex]
      }

      // Id is not relative to task, unlike index
      const itemId = itemStartIndex + itemInd + itemStartNum
      const timestamp = util.getItemTimestamp(itemExportMap[largestSensor])
      const newItem = convertItemToImport(
        itemExportMap[largestSensor].videoName as string,
        timestamp,
        itemExportMap,
        itemInd,
        itemId,
        attributeNameMap,
        attributeValueMap,
        categoryNameMap,
        tracking
      )

      if (tracking) {
        for (const label of Object.values(newItem.labels)) {
          if (isValidId(label.track) && !(label.track in trackMap)) {
            trackMap[label.track] = makeTrack(
              { type: label.type, id: label.track }, false)
          }
          trackMap[label.track].labels[label.item] = label.id
        }
      }
      maxOrder += Object.keys(newItem.labels).length

      itemsForTask.push(newItem)

      sensorMatchingIndices[largestSensor]++
    }

    // Update the num labels/shapes based on imports
    const taskStatus: TaskStatus = {
      maxOrder
    }

    const partialTask: Partial<TaskType> = {
      config,
      items: itemsForTask,
      status: taskStatus,
      sensors,
      tracks: trackMap
    }
    const task = makeTask(partialTask)
    tasks.push(task)
  }
  return Promise.resolve(tasks)
}
