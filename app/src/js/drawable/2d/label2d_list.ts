import _ from 'lodash'
import { policyFromString } from '../../common/track/track'
import { LabelTypeName, ShapeTypeName, TrackPolicyType } from '../../common/types'
import { makeState, makeTaskConfig } from '../../functional/states'
import { ConfigType, Label2DTemplateType, State } from '../../functional/types'
import { Context2D } from '../util'
import { Box2D } from './box2d'
import { CustomLabel2D } from './custom_label'
import { DrawMode, Label2D } from './label2d'
import { Node2D } from './node2d'
import { PathPoint2D } from './path_point2d'
import { Point2D } from './point2d'
import { Polygon2D } from './polygon2d'
import { Rect2D } from './rect2d'
import { Shape2D } from './shape2d'
import { Tag2D } from './tag2d'

/**
 * Make a new drawable label based on the label type
 * @param {string} labelType: type of the new label
 */
export function makeDrawableLabel2D (
  labelList: Label2DList,
  labelType: string,
  labelTemplates: { [name: string]: Label2DTemplateType }
): Label2D | null {
  if (labelType in labelTemplates) {
    return new CustomLabel2D(labelList, labelTemplates[labelType])
  }
  switch (labelType) {
    case LabelTypeName.BOX_2D:
      return new Box2D(labelList)
    case LabelTypeName.TAG:
      return new Tag2D(labelList)
    case LabelTypeName.POLYGON_2D:
      return new Polygon2D(labelList, true)
    case LabelTypeName.POLYLINE_2D:
      return new Polygon2D(labelList, false)
  }
  return null
}

/** Make drawable shape */
export function makeDrawableShape2D (
  shapeType: string
): Shape2D | null {
  switch (shapeType) {
    case ShapeTypeName.RECT:
      return new Rect2D()
    case ShapeTypeName.POINT_2D:
      return new Point2D()
    case ShapeTypeName.PATH_POINT_2D:
      return new PathPoint2D()
    case ShapeTypeName.NODE_2D:
      return new Node2D()
  }
  return null
}

/**
 * List of drawable labels
 * ViewController for the labels
 */
export class Label2DList {
  /** Currently highlighted label */
  public highlightedLabel: Label2D | null

  /** label id to label drawable map */
  private _labels: {[labelId: number]: Label2D}
  /** shape id to shape drawable map */
  private _shapes: {[shapeId: number]: Shape2D}
  /** list of the labels sorted by label order */
  private _labelList: Label2D[]
  /** selected label */
  private _selectedLabels: Label2D[]
  /** state */
  private _state: State
  /** label templates */
  private _labelTemplates: { [name: string]: Label2DTemplateType }
  /** callbacks */
  private _callbacks: Array<() => void>
  /** New labels to be committed */
  private _updatedLabels: Set<Label2D>
  /** updated shapes */
  private _updatedShapes: Set<Shape2D>
  /** task config */
  private _config: ConfigType
  /** next temporary shape id */
  private _temporaryShapeId: number
  /** selected item index */
  private _selectedItemIndex: number

  constructor () {
    this.highlightedLabel = null
    this._labels = {}
    this._shapes = {}
    this._labelList = []
    this._selectedLabels = []
    this._state = makeState()
    this._callbacks = []
    this._labelTemplates = {}
    this._updatedLabels = new Set()
    this._config = makeTaskConfig()
    this._updatedShapes = new Set()
    this._temporaryShapeId = -1
    this._selectedItemIndex = -1
  }

  /**
   * Access the drawable label by index
   */
  public get (index: number): Label2D {
    return this._labelList[index]
  }

  /** Subscribe callback for drawable update */
  public subscribe (callback: () => void) {
    this._callbacks.push(callback)
  }

  /** Unsubscribe callback for drawable update */
  public unsubscribe (callback: () => void) {
    const index = this._callbacks.indexOf(callback)
    if (index >= 0) {
      this._callbacks.splice(index, 1)
    }
  }

  /** Call when any drawable has been updated */
  public onDrawableUpdate (): void {
    for (const callback of this._callbacks) {
      callback()
    }
  }

  /** get shape by id */
  public getShape (id: number): Shape2D | null {
    if (id in this._shapes) {
      return this._shapes[id]
    }
    return null
  }

  /**
   * Get label by id
   * @param id
   */
  public getLabel (id: number): Label2D | null {
    if (id in this._labels) {
      return this._labels[id]
    }
    return null
  }

  /** get label list for state inspection */
  public get labelList (): Label2D[] {
    return this._labelList
  }

  /** get selectedLabel for state inspection */
  public get selectedLabels (): Label2D[] {
    return this._selectedLabels
  }

  /**
   * Get id's of selected labels
   */
  public get selectedLabelIds (): {[index: number]: number[]} {
    return this._state.user.select.labels
  }

  /**
   * Get current policy type
   */
  public get policyType (): TrackPolicyType {
    return policyFromString(
      this._state.task.config.policyTypes[this._state.user.select.policyType]
    )
  }

  /** Get current config */
  public get config (): Readonly<ConfigType> {
    return this._config
  }

  /** Get label template by name */
  public getLabelTemplate (name: string): Label2DTemplateType | null {
    if (name in this._labelTemplates) {
      return this._labelTemplates[name]
    }
    return null
  }

  /**
   * Draw label and control context
   * @param {Context2D} labelContext
   * @param {Context2D} controlContext
   * @param {number} ratio: ratio: display to image size ratio
   */
  public redraw (
      labelContext: Context2D,
      controlContext: Context2D,
      ratio: number,
      hideLabels?: boolean
    ): void {
    const labelsToDraw = (hideLabels) ?
      this._labelList.filter((label) => label.selected) :
      this._labelList
    labelsToDraw.forEach((v) => v.draw(labelContext, ratio, DrawMode.VIEW))
    labelsToDraw.forEach(
      (v) => v.draw(controlContext, ratio, DrawMode.CONTROL)
    )
    this.selectedLabels.forEach((label) => {
      if (label.labelId < 0) {
        label.draw(labelContext, ratio, DrawMode.VIEW)
        label.draw(controlContext, ratio, DrawMode.CONTROL)
      }
    })
  }

  /** Add temporary shape */
  public addTemporaryShape (shape: Shape2D) {
    this._shapes[this._temporaryShapeId] = shape
    const indexedShape = shape.toState()
    indexedShape.id = this._temporaryShapeId
    indexedShape.item = this._selectedItemIndex
    shape.updateState(indexedShape)
    this._temporaryShapeId--
    this.addUpdatedShape(shape)
    return shape
  }

  /** Get uncommitted labels */
  public get updatedShapes (): Readonly<Set<Readonly<Shape2D>>> {
    return this._updatedShapes
  }

  /** Add updated shape */
  public addUpdatedShape (shape: Shape2D) {
    this._updatedShapes.add(shape)
  }

  /**
   * update labels from the state
   */
  public updateState (state: State): void {
    this._state = state
    this._labelTemplates = state.task.config.label2DTemplates
    this._config = state.task.config

    const self = this
    const itemIndex = state.user.select.item
    const item = state.task.items[itemIndex]
    this._selectedItemIndex = itemIndex

    // remove any shapes not in the state
    self._shapes = Object.assign({} as typeof self._shapes,
        _.pick(self._shapes, _.keys(item.indexedShapes)))

    // update drawable shapes
    _.forEach(item.indexedShapes, (indexedShape, key) => {
      const shapeId = Number(key)
      if (!(shapeId in self._shapes)) {
        const newShape = makeDrawableShape2D(indexedShape.type)
        if (newShape) {
          self._shapes[shapeId] = newShape
        }
      }
      if (shapeId in self._shapes) {
        const drawableShape = self._shapes[shapeId]
        drawableShape.updateState(indexedShape)
      }
    })

    // remove any label not in the state
    self._labels = Object.assign({} as typeof self._labels,
        _.pick(self._labels, _.keys(item.labels)))

    // update drawable label values
    _.forEach(item.labels, (label, key) => {
      const labelId = Number(key)
      if (!(labelId in self._labels)) {
        const newLabel = makeDrawableLabel2D(
          this, label.type, this._labelTemplates
        )
        if (newLabel) {
          self._labels[labelId] = newLabel
        }
      }
      if (labelId in self._labels) {
        const drawableLabel = self._labels[labelId]
        if (!drawableLabel.editing) {
          drawableLabel.selected = false
          drawableLabel.updateState(item.labels[labelId])
        }
      }
    })

    // order the labels and assign order values
    self._labelList = _.sortBy(_.values(self._labels), [(label) => label.order])
    _.forEach(self._labelList,
      (l: Label2D, index: number) => { l.index = index })

    // Set selected labels
    if (!this.selectedLabels.some((label) => label.editing)) {
      this._selectedLabels = []
      const select = state.user.select
      if (select.item in select.labels) {
        for (const id of select.labels[select.item]) {
          if (id in this._labels) {
            this._selectedLabels.push(this._labels[id])
            this._labels[id].selected = true
          }
        }
      }
    }

    // Set label parents
    _.forEach(item.labels, (label) => {
      if (label.id in self._labels) {
        if (label.parent in self._labels) {
          self._labels[label.id].parent = self._labels[label.parent]
        } else {
          self._labels[label.id].parent = null
        }
      }
    })
  }

  /** Get uncommitted labels */
  public get updatedLabels (): Readonly<Set<Readonly<Label2D>>> {
    return this._updatedLabels
  }

  /** Push updated label to array */
  public addUpdatedLabel (label: Label2D) {
    this._updatedLabels.add(label)
  }

  /** Clear uncommitted label list */
  public clearUpdated () {
    this._updatedLabels.clear()
    this._updatedShapes.clear()
    this._temporaryShapeId = -1
  }
}
