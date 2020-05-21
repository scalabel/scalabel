import _ from 'lodash'
import { changeLabelsProps, linkLabels, mergeTracks, startLinkTrack, unlinkLabels } from '../../action/common'
import { selectLabels, unselectLabels } from '../../action/select'
import Session from '../../common/session'
import { Key } from '../../common/types'
import { getLinkedLabelIds } from '../../functional/common'
import { tracksOverlapping } from '../../functional/track'
import { IdType, State } from '../../functional/types'
import { Size2D } from '../../math/size2d'
import { Vector2D } from '../../math/vector2d'
import { commit2DLabels } from '../states'
import { Label2D } from './label2d'
import { makeDrawableLabel2D } from './label2d_list'

/**
 * List of drawable labels
 * ViewController for the labels
 */
export class Label2DHandler {
  /** Recorded state of last update */
  private _state: State
  /** highlighted label */
  private _highlightedLabel: Label2D | null
  /** The hashed list of keys currently down */
  private _keyDownMap: { [key: string]: boolean }
  /** index of currently selected item */
  private _selectedItemIndex: number

  constructor () {
    this._highlightedLabel = null
    this._state = Session.getState()
    this._keyDownMap = {}
    this._selectedItemIndex = -1
  }

  /** get highlightedLabel for state inspection */
  public get highlightedLabel (): Label2D | null {
    return this._highlightedLabel
  }

  /**
   * Process mouse down action
   * @param coord
   * @param labelIndex
   * @param handleIndex
   */
  public onMouseDown (
      coord: Vector2D, _labelIndex: number, handleIndex: number): boolean {
    if (!this.hasSelectedLabels() || !this.isEditingSelectedLabels()) {
      if (this._highlightedLabel) {
        this.selectHighlighted()
      } else {
        Session.dispatch(selectLabels(
          {}, -1, []
        ))
        Session.label2dList.selectedLabels.length = 0
        const state = this._state

        const label = makeDrawableLabel2D(
          Session.label2dList,
          state.task.config.labelTypes[state.user.select.labelType],
          state.task.config.label2DTemplates
        )
        if (label) {
          label.initTemp(state, coord)
          Session.label2dList.selectedLabels.push(label)
          Session.label2dList.labelList.push(label)
        }

        this._highlightedLabel = label
      }
    }
    if (this.hasSelectedLabels() &&
        !this.isKeyDown(Key.META) && !this.isKeyDown(Key.CONTROL)) {
      for (const label of Session.label2dList.selectedLabels) {
        if (label !== this._highlightedLabel) {
          label.setHighlighted(true, 0)
        }
        label.onMouseDown(coord, handleIndex)
      }
    }
    return true
  }

  /**
   * Process mouse up action
   * @param coord
   * @param labelIndex
   * @param handleIndex
   */
  public onMouseUp (
      coord: Vector2D, _labelIndex: number, _handleIndex: number): void {
    if (this.hasSelectedLabels() && !this.isKeyDown(Key.META)) {
      const labelsToRemove: Label2D[] = []
      Session.label2dList.selectedLabels.forEach((selectedLabel) => {
        selectedLabel.onMouseUp(coord)
        if (selectedLabel !== this._highlightedLabel) {
          selectedLabel.setHighlighted(false)
        }
        if (!selectedLabel.isValid() && !selectedLabel.editing) {
          labelsToRemove.push(selectedLabel)
        }
      })
      commit2DLabels([...Session.label2dList.updatedLabels.values()])
      Session.label2dList.clearUpdatedLabels()

      for (const label of labelsToRemove) {
        const labelListIndex = Session.label2dList.labelList.indexOf(label)
        if (labelListIndex >= 0) {
          Session.label2dList.labelList.splice(labelListIndex, 1)
        }
        const selectedIndex = Session.label2dList.selectedLabels.indexOf(label)
        if (selectedIndex >= 0) {
          Session.label2dList.selectedLabels.splice(selectedIndex, 1)
        }
      }
    }
  }

  /**
   * Process mouse move action
   */
  public onMouseMove (
      coord: Vector2D, canvasLimit: Size2D,
      labelIndex: number, handleIndex: number): boolean {
    if (this.hasSelectedLabels() && this.isEditingSelectedLabels()) {
      for (const label of Session.label2dList.selectedLabels) {
        label.onMouseMove(coord, canvasLimit, labelIndex, handleIndex)
        label.setManual()
      }
      return true
    } else {
      if (labelIndex >= 0) {
        if (!this._highlightedLabel) {
          this._highlightedLabel = Session.label2dList.labelList[labelIndex]
        }
        if (this._highlightedLabel.index !== labelIndex) {
          this._highlightedLabel.setHighlighted(false)
          this._highlightedLabel = Session.label2dList.labelList[labelIndex]
        }
        this._highlightedLabel.setHighlighted(true, handleIndex)
      } else if (this._highlightedLabel !== null) {
        this._highlightedLabel.setHighlighted(false)
        this._highlightedLabel = null
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
        if (this.isKeyDown(Key.CONTROL)) {
          // track link mode
          Session.dispatch(startLinkTrack())
        } else {
          // linking
          this.linkLabels()
        }
        break
      case Key.L_UP:
        // unlinking
        this.unlinkLabels()
        break
      case Key.ARROW_UP:
        // Only change order when one label is selected
        if (Session.label2dList.selectedLabels.length === 1) {
          const label = Session.label2dList.selectedLabels[0]
          this.swapOrders(label.index, label.index + 1)
        }
        break
      case Key.ARROW_DOWN:
        if (Session.label2dList.selectedLabels.length === 1) {
          const label = Session.label2dList.selectedLabels[0]
          this.swapOrders(label.index, label.index - 1)
        }
        break
      case Key.B_LOW:
      case Key.B_UP:
        if (Session.label2dList.selectedLabels.length === 1) {
          const selectedLabel = Session.label2dList.selectedLabels[0]
          this.changeLabelOrder(selectedLabel.index, 0)
        }
        break
      case Key.F_LOW:
      case Key.F_UP:
        if (Session.label2dList.selectedLabels.length === 1) {
          const selectedLabel = Session.label2dList.selectedLabels[0]
          this.changeLabelOrder(
            selectedLabel.index, Session.label2dList.labelList.length - 1
          )
        }
        break
      case Key.ENTER:
        if (this._state.session.trackLinking) {
          this.mergeTracks()
        }
        break
    }
  }

  /** Update state */
  public updateState (state: State) {
    this._state = state
    if (this._selectedItemIndex !== state.user.select.item) {
      this._highlightedLabel = null
    }
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

  /** returns whether selectedLabels is empty */
  private hasSelectedLabels (): boolean {
    return Session.label2dList.selectedLabels.length !== 0
  }

  /** returns whether selectedLabels is editing */
  private isEditingSelectedLabels (): boolean {
    for (const label of Session.label2dList.selectedLabels) {
      if (label.editing) {
        return true
      }
    }
    return false
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
    if (this._highlightedLabel !== null) {
      const item = this._state.task.items[this._state.user.select.item]
      const labelIds = getLinkedLabelIds(item, this._highlightedLabel.labelId)
      const highlightedAlreadySelected =
        Session.label2dList.selectedLabels.includes(
          this._highlightedLabel
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
            this._highlightedLabel.category[0],
            this._highlightedLabel.attributes,
            true
          ))
        }
      } else if (!highlightedAlreadySelected) {
        Session.dispatch(selectLabels(
          Session.label2dList.selectedLabelIds,
          this._selectedItemIndex,
          labelIds,
          this._highlightedLabel.category[0],
          this._highlightedLabel.attributes
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

  /** swap label orders, given label indices */
  private swapOrders (index1: number, index2: number) {
    // Check that indices are valid
    if (
      index1 >= 0 &&
      index2 >= 0 &&
      index1 < Session.label2dList.labelList.length &&
      index2 < Session.label2dList.labelList.length
    ) {
      const label1 = Session.label2dList.get(index1)
      const label2 = Session.label2dList.get(index2)
      Session.dispatch(changeLabelsProps(
        [this._selectedItemIndex],
        [[label1.labelId, label2.labelId]],
        [[{ order: label2.order }, { order: label1.order } ]]
      ))
    }
  }

  /** move label to nth position */
  private changeLabelOrder (index: number, newPosition: number) {
    const labels = Session.label2dList.labelList
    if (
      index >= 0 &&
      index < labels.length &&
      newPosition >= 0 &&
      newPosition < labels.length &&
      index !== newPosition
    ) {
      const start = Math.min(index, newPosition)
      const end = Math.max(index, newPosition)
      const labelIds: IdType[] = []
      const props = []
      for (let i = start; i <= end; i++) {
        labelIds.push(labels[i].labelId)
        props.push({ order: labels[i].order })
      }
      if (index < newPosition) {
        const labelId = labelIds.shift()
        if (labelId !== undefined) {
          labelIds.push(labelId)
        }
      } else {
        const labelId = labelIds.pop()
        if (labelId !== undefined) {
          labelIds.unshift(labelId)
        }
      }
      Session.dispatch(changeLabelsProps(
        [this._selectedItemIndex],
        [labelIds],
        [props]
      ))
    }
  }

  /**
   * Merge different tracks
   */
  private mergeTracks () {
    const selectedLabels = this._state.user.select.labels
    const selectedTracks = Object.entries(selectedLabels)
                             .map(([item, label]) =>
                             this._state.task.items[+item]
                              .labels[label[0]].track)
    const tracks = selectedTracks
                    .map((trackId) => this._state.task.tracks[trackId])
    if (!tracksOverlapping(tracks)) {
      Session.dispatch(mergeTracks(selectedTracks))
    }
  }
}
