import * as yaml from "js-yaml"
import _ from "lodash"

import {
  getInstructionUrl,
  getPageTitle,
  getTracking,
  index2str
} from "../common/util"
import { ItemTypeName, LabelTypeName } from "../const/common"
import { FormField } from "../const/project"
import {
  isValidId,
  makeSensor,
  makeTask,
  makeTrack
} from "../functional/states"
import {
  DatasetExport,
  ItemExport,
  ItemGroupExport,
  LabelExport
} from "../types/export"
import { CreationForm, FormFileData, Project } from "../types/project"
import {
  Attribute,
  Category,
  ConfigType,
  ItemType,
  Label2DTemplateType,
  SensorType,
  TaskStatus,
  TaskType,
  TrackIdMap
} from "../types/state"
import * as defaults from "./defaults"
import { convertItemToImport } from "./import"
import { ProjectStore } from "./project_store"
import { Storage } from "./storage"
import * as util from "./util"
import { mergeNearbyVertices, polyIsComplex } from "../math/polygon2d"
import Logger from "./logger"

/**
 * convert fields to form and validate input
 * if invalid input is found, error is returned to user via alert
 *
 * @param projectStore
 */
export async function parseForm(
  fields: { [key: string]: string },
  projectStore: ProjectStore
): Promise<CreationForm> {
  // Check that required fields were entered
  let projectName = fields[FormField.PROJECT_NAME]
  if (projectName === "") {
    throw Error("Please create a project name")
  } else {
    projectName = util.parseProjectName(projectName)
  }

  const itemType = fields[FormField.ITEM_TYPE]
  if (itemType === "") {
    throw Error("Please choose an item type")
  }

  const labelType = fields[FormField.LABEL_TYPE]
  if (labelType === "") {
    throw Error("Please choose a label type")
  }

  // Task size is not required for videos
  let taskSize = 1 // Video case
  if (fields[FormField.ITEM_TYPE] !== ItemTypeName.VIDEO) {
    if (fields[FormField.TASK_SIZE] === "") {
      throw Error("Please specify a task size")
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
    throw Error("Project name already exists.")
  }
  const demoMode = fields[FormField.DEMO_MODE] === "true"
  const form = util.makeCreationForm(
    projectName,
    itemType,
    labelType,
    pageTitle,
    taskSize,
    instructionUrl,
    demoMode
  )
  return form
}

/**
 * Parses item, category, and attribute files from paths
 *
 * @param storage
 * @param labelType
 * @param itemsRequired
 */
export async function parseFiles(
  storage: Storage,
  labelType: string,
  files: { [key: string]: string },
  itemsRequired: boolean
): Promise<FormFileData> {
  const items = parseItems(storage, files, itemsRequired)

  const categories: Promise<Category[]> = readConfig(
    storage,
    _.get(files, FormField.CATEGORIES),
    getDefaultCategories(labelType)
  )

  const sensors: Promise<SensorType[]> = readConfig(
    storage,
    _.get(files, FormField.SENSORS),
    []
  )

  const templates: Promise<Label2DTemplateType[]> = readConfig(
    storage,
    _.get(files, FormField.LABEL_SPEC),
    []
  )

  const attributes = readConfig(
    storage,
    _.get(files, FormField.ATTRIBUTES),
    getDefaultAttributes(labelType)
  )

  return await Promise.all([
    items,
    sensors,
    templates,
    attributes,
    categories
  ]).then(
    (
      result: [
        [Array<Partial<ItemExport>>, Array<Partial<ItemGroupExport>>],
        SensorType[],
        Label2DTemplateType[],
        Attribute[],
        Category[]
      ]
    ) => {
      return {
        items: result[0][0],
        itemGroups: result[0][1],
        sensors: result[1],
        templates: result[2],
        attributes: result[3],
        categories: result[4]
      }
    }
  )
}

/**
 * Parses item, category, and attribute files from paths
 *
 * @param storage
 * @param labelType
 */
export async function parseSingleFile(
  storage: Storage,
  labelType: string,
  files: { [key: string]: string }
): Promise<FormFileData> {
  const dataset: Promise<DatasetExport> = readConfig(
    storage,
    _.get(files, FormField.DATASET),
    getDefaultDataset(labelType)
  )

  return await dataset.then((dataset: DatasetExport) => {
    const categories: Category[] = []
    dataset.config.categories.forEach((category) =>
      categories.push({ name: category })
    )
    return {
      items: dataset.frames as Array<Partial<ItemExport>>,
      itemGroups: dataset.frameGroups !== undefined ? dataset.frameGroups : [],
      sensors:
        dataset.config.sensors !== undefined ? dataset.config.sensors : [],
      templates: [],
      attributes: dataset.config.attributes as Attribute[],
      categories: categories
    }
  })
}

/**
 * Read the config file, for example items or attributes
 * Can be in json or yaml format
 * If the path is undefined or empty, use the default
 *
 * @param storage
 * @param filePath
 * @param defaultValue
 */
export async function readConfig<T>(
  storage: Storage,
  filePath: string | undefined,
  defaultValue: T
): Promise<T> {
  if (filePath === undefined) {
    return defaultValue
  }

  const file = await storage.load(filePath)
  try {
    const fileData = (yaml.load(file, { json: true }) as unknown) as T
    return fileData
  } catch {
    throw new Error(`Improper formatting for file: ${filePath}`)
  }
}

/**
 * Get default categories if they weren't provided
 *
 * @param labelType
 */
function getDefaultCategories(labelType: string): Category[] {
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
 * Get default attributes if they weren't provided
 *
 * @param labelType
 */
function getDefaultAttributes(labelType: string): Attribute[] {
  switch (labelType) {
    case LabelTypeName.BOX_2D:
      return defaults.box2DAttributes
    default:
      return defaults.dummyAttributes
  }
}

/**
 * Get default dataset if it wasn't provided
 *
 * @param labelType
 */
function getDefaultDataset(labelType: string): DatasetExport {
  const dataset: DatasetExport = {
    frames: [],
    config: {
      attributes: getDefaultAttributes(labelType),
      categories: []
    }
  }
  return dataset
}

/**
 * Load from items file, grouped by video name
 *
 * @param storage
 * @param itemsRequired
 */
export async function parseItems(
  storage: Storage,
  files: { [key: string]: string },
  itemsRequired: boolean
): Promise<[Array<Partial<ItemExport>>, Array<Partial<ItemGroupExport>>]> {
  if (FormField.ITEMS in files) {
    let items = await readConfig<Array<Partial<ItemExport>> | DatasetExport>(
      storage,
      files[FormField.ITEMS],
      []
    )
    let itemGroups: Array<Partial<ItemGroupExport>> = []
    if ("frameGroups" in items) {
      itemGroups = items.frameGroups as Array<Partial<ItemGroupExport>>
    }
    if ("frames" in items) {
      items = items.frames
    }
    return [items, itemGroups]
  } else {
    if (itemsRequired) {
      throw new Error("No item file.")
    } else {
      return [[], []]
    }
  }
}

/**
 * Marshal data into project format
 *
 * @param form
 * @param formFileData
 */
export async function createProject(
  form: CreationForm,
  formFileData: FormFileData
): Promise<Project> {
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
    categories: getLeafCategories(formFileData.categories),
    treeCategories: formFileData.categories,
    attributes: formFileData.attributes,
    taskId: "",
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
      itemExport.videoName = ""
    }
  })

  const sensors: { [id: number]: SensorType } = {}
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
    itemGroups: formFileData.itemGroups,
    sensors
  }
  return Promise.resolve(project)
}

/**
 * Get leaf categories
 *
 * @param categoires
 */
function getLeafCategories(categoires: Category[]): string[] {
  let leafCategoires: string[] = []
  for (const category of categoires) {
    if (category.subcategories !== undefined) {
      leafCategoires = leafCategoires.concat(
        getLeafCategories(category.subcategories)
      )
    } else {
      leafCategoires.push(category.name)
    }
  }
  return leafCategoires
}

/**
 * Create two maps for quick lookup of attribute data
 *
 * @param configAttributes the attributes from config file
 * first RV: map from attribute name to attribute and its index
 * second RV: map from attribute value to its index within that attribute
 */
function getAttributeMaps(
  configAttributes: Attribute[]
): [{ [key: string]: [number, Attribute] }, { [key: string]: number }] {
  const attributeNameMap: { [key: string]: [number, Attribute] } = {}
  const attributeValueMap: { [key: string]: number } = {}
  for (let attrInd = 0; attrInd < configAttributes.length; attrInd++) {
    const configAttribute = configAttributes[attrInd]
    // Map attribute name to its index and its value
    attributeNameMap[configAttribute.name] = [attrInd, configAttribute]
    // Map attribute values to their indices (if its a list)
    if (configAttribute.type === "list") {
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
 *
 * @param configCategories the categories from config file
 * returns a map from category value to its index
 */
function getCategoryMap(configCategories: string[]): { [key: string]: number } {
  const categoryNameMap: { [key: string]: number } = {}
  for (let catInd = 0; catInd < configCategories.length; catInd++) {
    // Map category names to their indices
    const category = configCategories[catInd]
    categoryNameMap[category] = catInd
  }
  return categoryNameMap
}

/**
 * Create a map for quick lookup of items by their name
 *
 * @param items
 */
function getItemNameMap(
  items: Array<Partial<ItemExport>>
): { [key: string]: number } {
  const itemNameMap: { [key: string]: number } = {}
  for (let itemInd = 0; itemInd < items.length; itemInd++) {
    const item = items[itemInd]
    let itemName: string = ""
    item.name !== undefined
      ? (itemName = item.name)
      : (itemName = item.url as string)
    itemNameMap[itemName] = itemInd
  }
  return itemNameMap
}

/**
 * Filter invalid items, condition depends on whether labeling fusion data
 * Items are in export format
 *
 * @param items
 * @param itemType
 */
function filterInvalidItems(
  items: Array<Partial<ItemExport>>,
  itemType: string,
  sensors: { [id: number]: SensorType }
): Array<Partial<ItemExport>> {
  if (itemType === ItemTypeName.FUSION) {
    return items.filter(
      (itemExport) =>
        itemExport.dataType === undefined &&
        itemExport.sensor !== undefined &&
        itemExport.timestamp !== undefined &&
        itemExport.sensor in sensors
    )
  } else {
    return items.filter(
      (itemExport) =>
        itemExport.dataType === undefined || itemExport.dataType === itemType
    )
  }
}

/**
 * Make item groups
 *
 * @param items
 * @param itemGroups
 */
function makeItemGroups(
  items: Array<Partial<ItemExport>>,
  itemGroups?: Array<Partial<ItemGroupExport>>
): Array<Partial<ItemGroupExport>> {
  if (itemGroups !== undefined && itemGroups.length > 0) {
    return itemGroups
  } else {
    const newItemGroups: Array<Partial<ItemGroupExport>> = []
    for (const item of items) {
      const itemSensor = item.sensor as number
      let itemName = item.url as string
      if (item.name !== undefined) {
        itemName = item.name
      }
      const newItemGroup: Partial<ItemGroupExport> = {
        timestamp: util.getItemTimestamp(item),
        frames: { [itemSensor]: itemName },
        videoName: item.videoName
      }
      newItemGroups.push(newItemGroup)
    }
    return newItemGroups
  }
}

/**
 * If the imported items have polygon annotations,
 * check whether the polygon have intersections. Filter them and throw a warning
 *
 * @param project
 */
export function filterIntersectedPolygonsInProject(
  project: Project
): [Project, string] {
  const items = project.items
  let msg: string = ""
  let numberOfIntersections = 0

  const newItems = items.map((item) => {
    let newItem: Partial<ItemExport> = item
    if (item.labels !== undefined) {
      const filteredLabels: LabelExport[] = []
      item.labels.forEach((label) => {
        if (label.poly2d !== undefined && label.poly2d !== null) {
          label.poly2d.forEach((poly) => {
            // If it is a polyline label, do not check intersection
            if (!poly.closed) {
              filteredLabels.push(label)
            } else {
              // This is a workaround for importing bdd100k labels.
              // Its polygon may contain vertices that is very close (<1)
              // And the intersection there always appear under this situation
              // So we merge them first to avoid intersection
              poly.vertices = mergeNearbyVertices(poly.vertices, 1)
              // Check whether the polygon have intersections
              const intersectionData = polyIsComplex(poly.vertices)
              if (intersectionData.length > 0) {
                numberOfIntersections += intersectionData.length
                intersectionData.forEach((seg) => {
                  msg = `Image url: ${
                    item.url !== undefined ? item.url.toString() : ""
                  }\n`
                  msg += `polygon ID: ${label.id.toString()}\n`
                  msg += `Segment1: (${seg[0]}, ${seg[1]}, ${seg[2]}, ${seg[3]})\n`
                  msg += `Segment2: (${seg[4]}, ${seg[5]}, ${seg[6]}, ${seg[7]})\n`
                  msg += `\n`
                })
              } else {
                filteredLabels.push(label)
              }
            }
          })
        } else {
          filteredLabels.push(label)
        }
      })
      newItem = {
        ...newItem,
        labels: filteredLabels
      }
    }
    return newItem
  })

  if (numberOfIntersections > 0) {
    msg =
      `Found and filtered${numberOfIntersections} polygon intersection(s)!\n` +
      msg +
      "Please check your data."
    Logger.warning(msg)
  }

  project.items = newItems
  return [project, msg]
}

/**
 * Partitions the item into tasks
 * Returns list of task indices in format [start, stop) for every task
 *
 * @param itemGroups
 * @param tracking
 * @param taskSize
 */
function partitionItemsIntoTasks(
  itemGroups: Array<Partial<ItemGroupExport>>,
  tracking: boolean,
  taskSize: number
): number[] {
  const taskGroupIndices: number[] = []
  if (tracking) {
    // Partition by video name
    let prevVideoName: string
    itemGroups.forEach((value, index) => {
      if (value.videoName !== undefined) {
        if (value.videoName !== prevVideoName) {
          taskGroupIndices.push(index)
          prevVideoName = value.videoName
        }
      }
    })
    taskGroupIndices.push(itemGroups.length)
  } else {
    // If we have multiple sensors,
    // we should make those belong to the same group into the same task
    let currentSize = 0
    taskGroupIndices.push(0)
    for (let i = 0; i < itemGroups.length; i += 1) {
      currentSize += Object.keys(
        itemGroups[i].frames as { [id: number]: string }
      ).length
      if (currentSize >= taskSize) {
        taskGroupIndices.push(i + 1)
        currentSize = 0
      }
    }
    if (currentSize > 0) {
      taskGroupIndices.push(itemGroups.length)
    }
  }
  return taskGroupIndices
}

/**
 * Split project into tasks
 * Each consists of the task portion of a front  end state
 * Task and item start number are used if other tasks/items already exist
 *
 * @param project
 * @param projectStore
 * @param taskStartNum
 * @param itemStartNum
 * @param returnTask
 */
export async function createTasks(
  project: Project,
  projectStore?: ProjectStore,
  taskStartNum: number = 0,
  itemStartNum: number = 0,
  returnTask: boolean = false
): Promise<TaskType[]> {
  const sensors = project.sensors
  const { itemType, taskSize, tracking } = project.config

  const items = filterInvalidItems(project.items, itemType, sensors)

  // Update sensor info
  if (itemType !== ItemTypeName.FUSION) {
    if (Object.keys(sensors).length === 0) {
      sensors[-1] = makeSensor(-1, "default", itemType)
    }
    let maxSensorId = Math.max(
      ...Object.keys(sensors).map((key) => Number(key))
    )
    for (const itemExport of items) {
      if (itemExport.dataType !== undefined) {
        sensors[maxSensorId + 1] = makeSensor(
          maxSensorId,
          "",
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

  const itemGroups = makeItemGroups(items, project.itemGroups)
  const itemGroupIndices = partitionItemsIntoTasks(
    itemGroups,
    tracking,
    taskSize
  )

  /* create quick lookup dicts for conversion from export type
   * to external type for attributes/categories
   * this avoids lots of indexof calls which slows down creation */
  const [attributeNameMap, attributeValueMap] = getAttributeMaps(
    project.config.attributes
  )
  const categoryNameMap = getCategoryMap(project.config.categories)
  const tasks: TaskType[] = []

  const itemNameMap = getItemNameMap(items)
  for (
    let taskIndex = 0;
    taskIndex < itemGroupIndices.length - 1;
    taskIndex++
  ) {
    const itemGroupStartIndex = itemGroupIndices[taskIndex]
    const itemGroupEndIndex = itemGroupIndices[taskIndex + 1]
    const taskItemGroups = itemGroups.slice(
      itemGroupStartIndex,
      itemGroupEndIndex
    )

    const realTaskSize = taskItemGroups.length

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
      const itemExportMap: { [id: number]: Partial<ItemExport> } = {}
      const taskItemGroup = taskItemGroups[itemInd]
      const taskItemDict = taskItemGroup.frames as { [id: number]: string }
      for (const key of Object.keys(taskItemDict)) {
        const sensorId = Number(key)
        const taskItemName = taskItemDict[sensorId]
        const taskItemInd = itemNameMap[taskItemName]
        itemExportMap[sensorId] = items[taskItemInd]
      }

      // Id is not relative to task, unlike index
      const itemId = itemGroupStartIndex + itemInd + itemStartNum
      const timestamp = util.getItemTimestamp(taskItemGroup)
      const newItem = convertItemToImport(
        taskItemGroup.videoName as string,
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
              { type: label.type, id: label.track },
              false
            )
          }
          trackMap[label.track].labels[label.item] = label.id
        }
      }
      maxOrder += Object.keys(newItem.labels).length

      itemsForTask.push(newItem)
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
    if (projectStore !== undefined) {
      await projectStore.saveTasks([task])
    }

    if (returnTask) {
      tasks.push(task)
    }
  }
  return Promise.resolve(tasks)
}
