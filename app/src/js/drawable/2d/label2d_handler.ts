import _ from 'lodash'
import { linkLabels, unlinkLabels } from '../../action/common'
import { selectLabels, unselectLabels } from '../../action/select'
import Session from '../../common/session'
import { Key } from '../../common/types'
import { getLinkedLabelIds } from '../../functional/common'
import { State } from '../../functional/types'
import { Size2D } from '../../math/size2d'
import { Vector2D } from '../../math/vector2d'
import { commitLabels } from '../states'
import { getColorById } from '../util'
import { makeDrawableLabel2D } from './label2d_list'

const MOUSE_MOVE_THRESHOLD = 5

/**
 * List of drawable labels
 * ViewController for the labels
 */
export class Label2DHandler {
  /** Recorded state of last update */
  private _state: State
  /** The hashed list of keys currently down */
  private _keyDownMap: { [key: string]: boolean }
  /** index of currently selected item */
  private _selectedItemIndex: number
  /** Set if mouse has moved */
  private _mouseMoved: boolean
  /** Current mouse position */
  private _mouseCoord: Vector2D
  /** Set if mouse is down */
  private _mouseDown: boolean
  /** Handle of selected label */
  private _selectedHandle: number

  constructor () {
    this._state = Session.getState()
    this._keyDownMap = {}
    this._selectedItemIndex = -1
    this._mouseMoved = false
    this._mouseCoord = new Vector2D()
    this._mouseDown = false
    this._selectedHandle = -1
  }

  /**
   * Process mouse down action
   * @param coord
   * @param labelIndex
   * @param handleIndex
   */
  public onMouseDown (coord: Vector2D): boolean {
    if (!this.editingSelectedLabels()) {
      if (Session.label2dList.highlightedLabel) {
        this.selectHighlighted()
        if (this.isKeyDown(Key.META) || this.isKeyDown(Key.CONTROL)) {
          return true
        }
      } else if (!this.isKeyDown(Key.META) && !this.isKeyDown(Key.CONTROL)) {
        Session.dispatch(selectLabels({}, -1, []))
        Session.label2dList.selectedLabels.length = 0
        const state = this._state

        const label = makeDrawableLabel2D(
          Session.label2dList,
          state.task.config.labelTypes[state.user.select.labelType],
          state.task.config.label2DTemplates
        )
        if (label) {
          const status = state.task.status
          const color = getColorById(
            status.maxLabelId + 1,
            (Session.tracking) ? status.maxTrackId + 1 : -1
          )
          label.initTemp(
            status.maxOrder + 1,
            state.user.select.item,
            [state.user.select.category],
            state.user.select.attributes,
            color,
            coord,
            state.task.config.labelTypes[state.user.select.labelType]
          )
          Session.label2dList.selectedLabels.push(label)
        }

        Session.label2dList.highlightedLabel = label
      }
    }
    if (!this.isKeyDown(Key.META) && !this.isKeyDown(Key.CONTROL)) {
      this._mouseCoord.set(coord.x, coord.y)
      this._mouseDown = true
    }
    return false
  }

  /**
   * Process mouse up action
   * @param coord
   * @param labelIndex
   * @param handleIndex
   */
  public onMouseUp (
    coord: Vector2D, _labelIndex: number, _handleIndex: number
  ): void {
    if (this._mouseDown && !this._mouseMoved) {
      for (const label of Session.label2dList.selectedLabels) {
        const consumed = label.click(coord)
        if (consumed) {
          label.editing = true
        }
      }
    }
    this._mouseDown = false
    this._mouseMoved = false
    commitLabels(
      [...Session.label2dList.updatedLabels.values()],
      [...Session.label2dList.updatedShapes.values()]
    )
    Session.label2dList.clearUpdated()
  }

  /**
   * Process mouse move action
   */
  public onMouseMove (
      coord: Vector2D, canvasLimit: Size2D,
      labelIndex: number, handleIndex: number): boolean {
    const delta = coord.clone().subtract(this._mouseCoord)
    if (
      this._mouseDown &&
      delta.dot(delta) > MOUSE_MOVE_THRESHOLD * MOUSE_MOVE_THRESHOLD
    ) {
      this._mouseMoved = true
      if (this._selectedHandle >= 0 && Session.label2dList.highlightedLabel) {
        const consumed =
          Session.label2dList.highlightedLabel.drag(delta, canvasLimit)
        if (consumed) {
          Session.label2dList.highlightedLabel.editing = true
          Session.label2dList.highlightedLabel.setManual()
        }
      } else {
        for (const label of Session.label2dList.selectedLabels) {
          const consumed = label.drag(delta, canvasLimit)
          if (consumed) {
            label.editing = true
            label.setManual()
          }
        }
      }
      this._mouseCoord.set(coord.x, coord.y)
      return true
    } else if (!this._mouseDown) {
      this._selectedHandle = -1
      if (Session.label2dList.highlightedLabel) {
        Session.label2dList.highlightedLabel.setHighlighted(false)
        Session.label2dList.highlightedLabel =
          Session.label2dList.labelList[labelIndex]
      }
      if (labelIndex >= 0) {
        Session.label2dList.highlightedLabel =
          Session.label2dList.labelList[labelIndex]
        Session.label2dList.highlightedLabel.setHighlighted(true, handleIndex)
        if (Session.label2dList.highlightedLabel.selected) {
          this._selectedHandle = handleIndex
        }
      }
    }
    return false
  }

  /**
   * Handle keyboard down events
   * @param e
   */
  public onKeyDown (e: KeyboardEvent): void {
    this._keyDownMap[e.key] = true
    for (const selectedLabel of Session.label2dList.selectedLabels) {
      if (!selectedLabel.onKeyDown(e.key)) {
        Session.label2dList.labelList.splice(
          Session.label2dList.labelList.indexOf(
            Session.label2dList.selectedLabels[0]
          ),
          1
        )
        Session.label2dList.selectedLabels.length = 0
      }
    }
    switch (e.key) {
      case Key.L_LOW:
        // linking
        this.linkLabels()
        break
      case Key.L_UP:
        // unlinking
        this.unlinkLabels()
        break
    }
  }

  /** Update state */
  public updateState (state: State) {
    this._state = state
    this._selectedItemIndex = state.user.select.item
  }

  /**
   * Handle keyboard up events
   * @param e
   */
  public onKeyUp (e: KeyboardEvent): void {
    delete this._keyDownMap[e.key]
    for (const selectedLabel of Session.label2dList.selectedLabels) {
      selectedLabel.onKeyUp(e.key)
    }
  }

  /** returns whether selectedLabels is editing */
  private editingSelectedLabels (): boolean {
    return Session.label2dList.selectedLabels.some((label) => label.editing)
  }

  /**
   * Whether a specific key is pressed down
   * @param key - the key to check
   */
  private isKeyDown (key: Key): boolean {
    return this._keyDownMap[key]
  }

  /** Select highlighted label, if any */
  private selectHighlighted (): void {
    if (Session.label2dList.highlightedLabel !== null) {
      const item = this._state.task.items[this._state.user.select.item]
      const labelIds =
        getLinkedLabelIds(item, Session.label2dList.highlightedLabel.labelId)
      const highlightedAlreadySelected =
        Session.label2dList.selectedLabels.includes(
          Session.label2dList.highlightedLabel
        )
      if (this.isKeyDown(Key.CONTROL) || this.isKeyDown(Key.META)) {
        if (highlightedAlreadySelected) {
          Session.dispatch(unselectLabels(
            Session.label2dList.selectedLabelIds,
            this._selectedItemIndex,
            labelIds
          ))
        } else {
          Session.dispatch(selectLabels(
            Session.label2dList.selectedLabelIds,
            this._selectedItemIndex,
            labelIds,
            Session.label2dList.highlightedLabel.category[0],
            Session.label2dList.highlightedLabel.attributes,
            true
          ))
        }
      } else if (!highlightedAlreadySelected) {
        Session.dispatch(selectLabels(
          Session.label2dList.selectedLabelIds,
          this._selectedItemIndex,
          labelIds,
          Session.label2dList.highlightedLabel.category[0],
          Session.label2dList.highlightedLabel.attributes
        ))
      }
    }
  }

  /** link selected labels */
  private linkLabels (): void {
    const selectedLabelIdArray = _.map(
      Session.label2dList.selectedLabels, (label) => label.labelId)
    Session.dispatch(linkLabels(
      this._state.user.select.item, selectedLabelIdArray
    ))
  }

  /** unlink selected labels */
  private unlinkLabels (): void {
    const selectedLabelIdArray = _.map(
      Session.label2dList.selectedLabels, (label) => label.labelId)
    Session.dispatch(unlinkLabels(
      this._state.user.select.item, selectedLabelIdArray
    ))
  }
}
