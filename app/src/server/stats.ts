import _ from "lodash"

import { ProjectOptions, TaskOptions } from "../components/dashboard"
import { AttributeToolType } from "../const/common"
import { Project } from "../types/project"
import { Attribute, LabelType, TaskType } from "../types/state"

/**
 * Extract ProjectOption from a Project
 *
 * @param project
 */
export function getProjectOptions(project: Project): ProjectOptions {
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
 *
 * @param task
 */
export function countLabelsTask(task: TaskType): number {
  const numPerItem = task.items.map((item) => _.size(item.labels))
  return _.sum(numPerItem)
}

/**
 * Returns the number of items with at least 1 label in the task
 *
 * @param task
 */
export function countLabeledItemsTask(task: TaskType): number {
  const labeledItems = task.items.filter((item) => _.size(item.labels) > 0)
  return labeledItems.length
}

/**
 * Returns the total number of labels in the project
 *
 * @param tasks
 */
export function countLabelsProject(tasks: TaskType[]): number {
  return _.sum(tasks.map(countLabelsTask))
}

/**
 * Returns the number of items with at least 1 label in the project
 *
 * @param tasks
 */
export function countLabeledItemsProject(tasks: TaskType[]): number {
  return _.sum(tasks.map(countLabeledItemsTask))
}

/**
 * Extracts TaskOptions from a Task
 *
 * @param task
 */
export function getTaskOptions(task: TaskType): TaskOptions {
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
 *
 * @param tasks
 */
export function getNumItems(tasks: TaskType[]): number {
  const itemsPerTask = tasks.map((task) => task.items.length)
  return _.sum(itemsPerTask)
}

/**
 * Get the number of tasks with a submission
 *
 * @param tasks
 */
export function getNumSubmissions(tasks: TaskType[]): number {
  const submittedTasks = tasks.filter(
    (task) => task.progress.submissions.length > 0
  )
  return submittedTasks.length
}

/**
 * Initialize the label counts for each attribute to 0
 *
 * @param attributes list of possible attributes
 * @returns map of attribute counts initialized to 0
 */
function initAttributeStats(attributes: Attribute[]): AttributeStats {
  const attributesByName = _.keyBy(attributes, (attribute) => attribute.name)
  return _.mapValues(attributesByName, (attribute) => {
    if (attribute.toolType === AttributeToolType.SWITCH) {
      return {
        false: 0,
        true: 0
      }
    }
    return _.zipObject(
      attribute.values,
      _.times(attribute.values.length, _.constant(0))
    )
  })
}

/**
 * Initialize category stats
 *
 * @param categories list of possible categories
 * @param attributes list of possible attributes
 * @returns map of category and attribute counts initialized to 0
 */
function initCategoryStats(
  categories: string[],
  attributes: Attribute[]
): CategoryStats {
  const categoriesByName = _.keyBy(categories)
  return _.mapValues(categoriesByName, () => {
    return {
      count: 0,
      attribute: initAttributeStats(attributes)
    }
  })
}

/**
 * Get the value of an attribute given the index in the value list
 *
 * @param attribute
 * @param index
 */
function getAttributeValue(attribute: Attribute, index: number): string {
  if (attribute.toolType === AttributeToolType.SWITCH) {
    return index === 1 ? "true" : "false"
  }
  return attribute.values[index]
}

/**
 * Updates the attribute stats with given label
 * Modifies the stats in place
 *
 * @param stats
 * @param label
 * @param attributes
 */
function updateAttributeStats(
  stats: AttributeStats,
  label: LabelType,
  attributes: Attribute[]
): void {
  const attributeIndices = _.map(
    Object.keys(label.attributes),
    (attributeKey) => Number(attributeKey)
  )

  for (const attributeIndex of attributeIndices) {
    const attribute = attributes[attributeIndex]
    const values = label.attributes[attributeIndex].map((valueIndex) =>
      getAttributeValue(attribute, valueIndex)
    )

    for (const value of values) {
      stats[attribute.name][value] += 1
    }
  }
}

/**
 * Update the category stats with the given label
 *
 * @param stats the stats so far
 * @param label the new label
 * @param categories the list of possible categories
 * @param attributes the list of possible attributes
 * Modifies the stats in place
 */
function updateCategoryStats(
  stats: CategoryStats,
  label: LabelType,
  categories: string[],
  attributes: Attribute[]
): void {
  const categoryNames = label.category.map(
    (categoryIndex) => categories[categoryIndex]
  )
  for (const categoryName of categoryNames) {
    stats[categoryName].count += 1
    updateAttributeStats(stats[categoryName].attribute, label, attributes)
  }
}

/**
 * Get the stats breakdown for all labels
 *
 * @param tasks
 */
export function getLabelStats(tasks: TaskType[]): LabelStats {
  if (tasks.length === 0) {
    return { category: {}, attribute: {} }
  }

  const config = tasks[0].config
  const categories = config.categories
  const attributes = config.attributes

  const categoryStats = initCategoryStats(categories, attributes)
  const attributeStats = initAttributeStats(attributes)

  const items = _.flatMap(tasks, (task) => task.items)
  const labels = _.flatMap(items, (item) => Object.values(item.labels))

  for (const label of labels) {
    updateCategoryStats(categoryStats, label, categories, attributes)
    updateAttributeStats(attributeStats, label, attributes)
  }

  return {
    category: categoryStats,
    attribute: attributeStats
  }
}

/** the number of labels for each attribute type/value */
interface AttributeStats {
  [name: string]: { [value: string]: number }
}

/**
 * Stats for label counts by category
 */
interface CategoryStats {
  [name: string]: {
    /** the number of labels with the category */
    count: number
    /** the counts for each attribute within that category */
    attribute: AttributeStats
  }
}

/**
 * Stats for label counts by different metrics
 */
export interface LabelStats {
  /** counts for each  category */
  category: CategoryStats
  /** counts for each attribute */
  attribute: AttributeStats
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
  /** stats for the labels */
  labelStats: LabelStats
  /** the time of retrieval for the stats */
  timestamp: number
}
/**
 * Get the stats for a collection of tasks from a project
 *
 * @param tasks
 */
export function getProjectStats(tasks: TaskType[]): ProjectStats {
  return {
    numLabels: countLabelsProject(tasks),
    numLabeledItems: countLabeledItemsProject(tasks),
    numItems: getNumItems(tasks),
    numSubmittedTasks: getNumSubmissions(tasks),
    numTasks: tasks.length,
    labelStats: getLabelStats(tasks),
    timestamp: Date.now()
  }
}
