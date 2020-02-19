import { Fields, Files } from 'formidable'
import * as fs from 'fs-extra'
import * as yaml from 'js-yaml'
import _ from 'lodash'
import { ItemTypeName, LabelTypeName } from '../common/types'
import { ItemExport } from '../functional/bdd_types'
import { makeSensor, makeTask, makeTrack } from '../functional/states'
import {
  Attribute,
  ConfigType,
  ItemType,
  Label2DTemplateType,
  SensorType,
  TaskStatus,
  TaskType,
  TrackMapType
} from '../functional/types'
import { convertItemToImport } from './import'
import { ProjectStore } from './project_store'
import * as types from './types'
import * as util from './util'

/**
 * convert fields to form and validate input
 * if invalid input is found, error is returned to user via alert
 */
export async function parseForm (
  fields: Fields, projectStore: ProjectStore): Promise<types.CreationForm> {
  // Check that required fields were entered
  let projectName = fields[types.FormField.PROJECT_NAME] as string
  if (projectName === '') {
    throw(Error('Please create a project name'))
  } else {
    projectName = projectName.replace(' ', '_')
  }

  const itemType = fields[types.FormField.ITEM_TYPE] as string
  if (itemType === '') {
    throw(Error('Please choose an item type'))
  }

  const labelType = fields[types.FormField.LABEL_TYPE] as string
  if (labelType === '') {
    throw(Error('Please choose a label type'))
  }

  // Task size is not required for videos
  let taskSize = 1 // video case
  if (fields[types.FormField.ITEM_TYPE] !== ItemTypeName.VIDEO) {
    if (fields[types.FormField.TASK_SIZE] === '') {
      throw(Error('Please specify a task size'))
    } else {
      taskSize = parseInt(fields[types.FormField.TASK_SIZE] as string, 10)
    }
  }

  // Non-required fields
  const pageTitle = fields[types.FormField.PAGE_TITLE] as string
  const instructions = fields[types.FormField.INSTRUCTIONS_URL] as string

  // Ensure project name is not already in use
  const exists = await projectStore.checkProjectName(projectName)
  if (exists) {
    throw(Error('Project name already exists.'))
  }
  const demoMode = fields[types.FormField.DEMO_MODE] === 'true'
  const form = util.makeCreationForm(
    projectName, itemType, labelType, pageTitle, taskSize,
    instructions, demoMode
  )
  return form
}

/**
 * Parses item, category, and attribute files from form
 */
export async function parseFiles (labelType: string, files: Files)
  : Promise<types.FormFileData> {
  return Promise.all([
    parseItems(files),
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
      return types.defaultBoxCategories
    case LabelTypeName.POLYLINE_2D:
      return types.defaultPolyline2DCategories
    default:
      return []
  }
}

/**
 * Read categories from yaml file at path
 */
function readCategoriesFile (path: string): Promise<string[]> {
  return new Promise((resolve, reject) => {
    fs.readFile(path, 'utf8', (err: types.MaybeError, file: string) => {
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
  files: Files, labelType: string): Promise<string[]> {
  const categoryFile = files[types.FormField.CATEGORIES]
  if (util.formFileExists(categoryFile)) {
    return readCategoriesFile(categoryFile.path)
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
      return types.defaultBox2DAttributes
    default:
      return types.dummyAttributes
  }
}

/**
 * Read attributes from yaml file at path
 */
function readAttributesFile (path: string): Promise<Attribute[]> {
  return new Promise((resolve, reject) => {
    fs.readFile(path, 'utf8', (err: types.MaybeError, fileBytes: string) => {
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
  files: Files, labelType: string): Promise<Attribute[]> {
  const attributeFile = files[types.FormField.ATTRIBUTES]
  if (util.formFileExists(attributeFile)) {
    return readAttributesFile(attributeFile.path)
  } else {
    const defaultAttributes = getDefaultAttributes(labelType)
    return Promise.resolve(defaultAttributes)
  }
}

/**
 * Read items from yaml file at path
 */
function readItemsFile (path: string): Promise<Array<Partial<ItemExport>>> {
  return new Promise((resolve, reject) => {
    fs.readFile(path, 'utf8', (err: types.MaybeError, fileBytes: string) => {
      if (err) {
        reject(err)
        return
      }
      try {
        // might not have all fields defined, so use partial
        const items = yaml.load(fileBytes) as Array<Partial<ItemExport>>
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
export function parseItems (files: Files): Promise<Array<Partial<ItemExport>>> {
  const itemFile = files[types.FormField.ITEMS]
  if (util.formFileExists(itemFile)) {
    return readItemsFile(itemFile.path)
  } else {
    return Promise.reject(Error('No item file.'))
  }
}

/** Read sensors file */
function readSensorsFile (path: string): Promise<SensorType[]> {
  return new Promise((resolve, reject) => {
    fs.readFile(path, 'utf8', (err: types.MaybeError, fileBytes: string) => {
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
export function parseSensors (files: Files): Promise<SensorType[]> {
  const sensorsFile = files[types.FormField.SENSORS]
  if (util.formFileExists(sensorsFile)) {
    return readSensorsFile(sensorsFile.path)
  } else {
    return Promise.resolve([])
  }
}

/** Read sensors file */
function readTemplatesFile (path: string): Promise<Label2DTemplateType[]> {
  return new Promise((resolve, reject) => {
    fs.readFile(path, 'utf8', (err: types.MaybeError, fileBytes: string) => {
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
export function parseTemplates (files: Files): Promise<Label2DTemplateType[]> {
  const templatesFile = files[types.FormField.LABEL_SPEC]
  if (util.formFileExists(templatesFile)) {
    return readTemplatesFile(templatesFile.path)
  } else {
    return Promise.resolve([])
  }
}

/**
 * Marshal data into project format
 */
export function createProject (
  form: types.CreationForm,
  formFileData: types.FormFileData): Promise<types.Project> {

  const handlerUrl = util.getHandlerUrl(form.itemType, form.labelType)
  const bundleFile = util.getBundleFile(form.labelType)
  const [itemType, tracking] = util.getTracking(form.itemType)

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
    instructionPage: form.instructions,
    bundleFile,
    categories: formFileData.categories,
    attributes: formFileData.attributes,
    taskId: '',
    tracking,
    policyTypes: [],
    demoMode: form.demoMode,
    autosave: true
  }

  // ensure that all videonames are set to default if empty
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

  // With tracking, order by videoname lexicographically and split according
  // to videoname. It should be noted that a stable sort must be used to
  // maintain ordering provided in the image list file
  projectItems = _.sortBy(projectItems, [(item) => item.videoName])
  const project: types.Project = {
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
    // map attribute name to its index and its value
    attributeNameMap[configAttribute.name] = [attrInd, configAttribute]
    // map attribute values to their indices (if its a list)
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
    // map category names to their indices
    const category = configCategories[catInd]
    categoryNameMap[category] = catInd
  }
  return categoryNameMap
}

// /**
//  * gets the max of values and currMax
//  * @param values an array of numbers in string format
//  */
// function getMax (values: string[], oldMax: number): number {
//   const numericValues = values.map((value: string) => {
//     return parseInt(value, 10)
//   })
//   let currMax = -1
//   if (numericValues.length > 0) {
//     currMax = numericValues.reduce((a, b) => {
//       return Math.max(a, b)
//     })
//   }
//   return Math.max(currMax, oldMax)
// }

/**
 * Split project into tasks
 * Each consists of the task portion of a frontend state
 */
export function createTasks (project: types.Project): Promise<TaskType[]> {
  // Filter invalid items, condition depends on whether labeling fusion data
  const items = (project.config.itemType === ItemTypeName.FUSION) ?
    project.items.filter((itemExport) =>
      itemExport.dataType === undefined &&
      itemExport.sensor !== undefined &&
      itemExport.timestamp !== undefined &&
      itemExport.sensor in project.sensors
    ) :
    project.items.filter((itemExport) =>
      !itemExport.dataType ||
      itemExport.dataType === project.config.itemType
    )

  let taskSize = project.config.taskSize

  /* create quick lookup dicts for conversion from export type
   * to external type for attributes/categories
   * this avoids lots of indexof calls which slows down creation */
  const [attributeNameMap, attributeValueMap] = getAttributeMaps(
    project.config.attributes)
  const categoryNameMap = getCategoryMap(project.config.categories)

  const sensors: {[id: number]: SensorType} = project.sensors
  if (project.config.itemType !== ItemTypeName.FUSION) {
    sensors[-1] = makeSensor(-1, 'default', project.config.itemType)
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

  const tasks: TaskType[] = []
  // taskIndices contains each [start, stop) range for every task
  const taskIndices: number[] = []
  if (project.config.tracking) {
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
    for (let i = 0; i < items.length; i += taskSize) {
      taskIndices.push(i)
    }
  }
  taskIndices.push(items.length)
  let taskStartIndex: number
  let taskEndIndex: number
  for (let i = 0; i < taskIndices.length - 1; i ++) {
    taskStartIndex = taskIndices[i]
    taskEndIndex = taskIndices[i + 1]
    const taskItemsExport = items.slice(taskStartIndex, taskEndIndex)

    // Map from data source id to list of item exports
    const itemExportsBySensor:
      {[id: number]: Array<Partial<ItemExport>>} = {}
    for (const itemExport of taskItemsExport) {
      if (itemExport.sensor !== undefined) {
        if (!(itemExport.sensor in itemExportsBySensor)) {
          itemExportsBySensor[itemExport.sensor] = []
        }
        itemExportsBySensor[itemExport.sensor].push(itemExport)
      }
    }

    taskSize = 0
    let largestSensor = -1
    const sensorMatchingIndices: {[id: number]: number} = {}
    for (const key of Object.keys(itemExportsBySensor)) {
      const sensorId = Number(key)
      itemExportsBySensor[sensorId] = _.sortBy(
        itemExportsBySensor[sensorId],
        [(itemExport) => (itemExport.timestamp === undefined) ?
          0 : itemExport.timestamp]
      )
      taskSize = Math.max(
        taskSize, itemExportsBySensor[sensorId].length
      )
      if (taskSize === itemExportsBySensor[sensorId].length) {
        largestSensor = sensorId
      }
      sensorMatchingIndices[sensorId] = 0
    }

    /* assign task id,
     and update task size in case there aren't enough items */
    const config: ConfigType = {
      ...project.config,
      taskSize,
      taskId: util.index2str(i)
    }

    // based on the imported labels, compute max ids
    let maxLabelId = -1
    let maxShapeId = -1
    let maxTrackId = -1
    // max order is the total number of labels
    let maxOrder = 0

    // convert from export format to internal format
    const itemsForTask: ItemType[] = []
    const trackMap: TrackMapType = {}
    for (let itemInd = 0; itemInd < taskSize; itemInd += 1) {
      const timestampToMatch = itemExportsBySensor[largestSensor][
        sensorMatchingIndices[largestSensor]
      ].timestamp as number
      const itemExportMap: {[id: number]: Partial<ItemExport>} = {}
      for (const key of Object.keys(sensorMatchingIndices)) {
        const sensorId = Number(key)
        let newIndex = sensorMatchingIndices[sensorId]
        const itemExports = itemExportsBySensor[sensorId]
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

      // id is not relative to task, unlike index
      const itemId = taskStartIndex + itemInd
      const timestamp = (
        (itemExportMap[largestSensor].timestamp !== undefined) ?
          itemExportMap[largestSensor].timestamp : 0
      ) as number
      const [newItem, newMaxLabelId, newMaxShapeId] = convertItemToImport(
        itemExportMap[largestSensor].videoName as string,
        timestamp,
        itemExportMap,
        itemInd,
        itemId,
        attributeNameMap,
        attributeValueMap,
        categoryNameMap,
        maxLabelId,
        maxShapeId,
        project.config.tracking
      )

      if (project.config.tracking) {
        for (const label of Object.values(newItem.labels)) {
          if (label.track >= 0 && !(label.track in trackMap)) {
            trackMap[label.track] = makeTrack(label.track, label.type)
          }
          trackMap[label.track].labels[label.item] = label.id
          maxTrackId = Math.max(label.track + 1, maxTrackId)
        }
      }

      maxLabelId = newMaxLabelId
      maxShapeId = newMaxShapeId
      maxOrder += Object.keys(newItem.labels).length

      itemsForTask.push(newItem)

      sensorMatchingIndices[largestSensor]++
    }

    // update the num labels/shapes based on imports
    const taskStatus: TaskStatus = {
      maxLabelId,
      maxShapeId,
      maxOrder,
      maxTrackId
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
