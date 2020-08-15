import _ from 'lodash'
import { ProjectOptions, TaskOptions } from '../components/dashboard'
import { AttributeToolType } from '../const/common'
import { Project } from '../types/project'
import { Attribute, LabelType, TaskType } from '../types/state'

/**
 * Extract ProjectOption from a Project
 */
export function getProjectOptions (project: Project): ProjectOptions {
  return {
    name: project.config.projectName,
    itemType: project.config.itemType,
    labelTypes: project.config.labelTypes,
    taskSize: project.config.taskSize,
    numItems: project.items.length,
    numLeafCategories: project.config.categories.length,
    numAttributes: project.config.attributes.length
  }
}

/**
 * Returns the total number of labels in the task
 */
export function countLabelsTask (task: TaskType): number {
  const numPerItem = task.items.map((item) => _.size(item.labels))
  return _.sum(numPerItem)
}

/**
 * Returns the number of items with at least 1 label in the task
 */
export function countLabeledItemsTask (task: TaskType): number {
  const labeledItems = task.items.filter((item) => _.size(item.labels) > 0)
  return labeledItems.length
}

/**
 * Returns the total number of labels in the project
 */
export function countLabelsProject (tasks: TaskType[]): number {
  return _.sum(tasks.map(countLabelsTask))
}

/**
 * Returns the number of items with at least 1 label in the project
 */
export function countLabeledItemsProject (tasks: TaskType[]): number {
  return _.sum(tasks.map(countLabeledItemsTask))

}

/**
 * Extracts TaskOptions from a Task
 */
export function getTaskOptions (task: TaskType): TaskOptions {
  const numLabeledItems = countLabeledItemsTask(task)
  const numLabels = countLabelsTask(task)
  return {
    numLabeledItems: numLabeledItems.toString(),
    numLabels: numLabels.toString(),
    submissions: task.progress.submissions,
    handlerUrl: task.config.handlerUrl
  }
}

/**
 * Get the total number of items across all tasks
 */
export function getNumItems (tasks: TaskType[]) {
  const itemsPerTask = tasks.map((task) => task.items.length)
  return _.sum(itemsPerTask)
}

/**
 * Get the number of tasks with a submission
 */
export function getNumSubmissions (tasks: TaskType[]) {
  const submittedTasks = tasks.filter((task) =>
    task.progress.submissions.length > 0)
  return submittedTasks.length
}

/**
 * Initialize the label counts for each attribute to 0
 * @param attributes list of possible attributes
 * @returns map of attribute counts initialized to 0
 */
function initAttributeCount (attributes: Attribute[]): AttributeCounts {
  const attributesByName = _.keyBy(attributes, (attribute) => attribute.name)
  return _.mapValues(attributesByName,
    (attribute) => {
      if (attribute.toolType === AttributeToolType.SWITCH) {
        return {
          false: 0,
          true: 0
        }
      }
      return _.zipObject(
        attribute.values,
        _.times(attribute.values.length, _.constant(0)))
    }
  )
}

/**
 * Initialize classification stats
 * @param categories list of possible categories
 * @param attributes list of possible attributes
 * @return map of category and attribute counts initialized to 0
 */
function initClassificationStats (
  categories: string[], attributes: Attribute[]):
  ClassificationStats {
  const categoriesByName = _.keyBy(categories)
  return _.mapValues(categoriesByName,
    (_category) => {
      return {
        count: 0,
        attributeCounts: initAttributeCount(attributes)
      }
    })
}

/**
 * Get the value of an attribute given the index in the value list
 */
function getAttributeValue (attribute: Attribute, index: number) {
  if (attribute.toolType === AttributeToolType.SWITCH) {
    return index === 1 ? 'true' : 'false'
  }
  return attribute.values[index]
}

/**
 * Update the classification stats with the given label
 * @param stats the stats so far
 * @param label the new label
 * @param categories the list of possible categories
 * @param attributes the list of possible attributes
 */
function updateClassificationStats (
  stats: ClassificationStats, label: LabelType,
  categories: string[], attributes: Attribute[]): ClassificationStats {
  const result = stats
  for (const categoryIndex of label.category) {
    const categoryName = categories[categoryIndex]
    result[categoryName].count += 1
    for (const attributeKey of Object.keys(label.attributes)) {
      const attributeIndex = Number(attributeKey)
      const attr = attributes[attributeIndex]

      for (const attributeValueIndex of label.attributes[attributeIndex]) {
        const value = getAttributeValue(attr, attributeValueIndex)
        result[categoryName].attributeCounts[attr.name][value] += 1
      }
    }
  }
  return result
}

/**
 * Get the label breakdown by class
 */
export function getClassificationStats (
  tasks: TaskType[]): ClassificationStats {
  if (tasks.length === 0) {
    return {}
  }

  const config = tasks[0].config
  const categories = config.categories
  const attributes = config.attributes

  let result = initClassificationStats(categories, attributes)

  for (const task of tasks) {
    for (const item of task.items) {
      for (const label of Object.values(item.labels)) {
        result = updateClassificationStats(
          result, label, categories, attributes)
      }
    }
  }
  return result
}

/** the number of labels for each attribute type/value */
interface AttributeCounts {
  [name: string]: { [value: string]: number}
}

/**
 * Stats for a single category
 */
interface CategoryStats {
  /** the number of labels with the category */
  count: number
  /** the counts for each attribute within the category */
  attributeCounts: AttributeCounts
}

/**
 * Stats for all classification categories
 */
export interface ClassificationStats {
  [name: string]: CategoryStats
}

/**
 * Stats for all the tasks of a project
 */
interface ProjectStats {
  /** the total number of labels */
  numLabels: number
  /** the number of labeled items */
  numLabeledItems: number
  /** the total number of items */
  numItems: number
  /** the number of submitted tasks */
  numSubmittedTasks: number
  /** the total number of tasks */
  numTasks: number
  /** stats for the classification of the labels */
  classificationStats: ClassificationStats
  /** the time of retrieval for the stats */
  timestamp: number
}
/**
 * Get the stats for a collection of tasks from a project
 */
export function getProjectStats (tasks: TaskType[]): ProjectStats {
  return {
    numLabels: countLabelsProject(tasks),
    numLabeledItems: countLabeledItemsProject(tasks),
    numItems: getNumItems(tasks),
    numSubmittedTasks: getNumSubmissions(tasks),
    numTasks: tasks.length,
    classificationStats: getClassificationStats(tasks),
    timestamp: Date.now()
  }
}
