import Divider from '@material-ui/core/Divider'
import List from '@material-ui/core/List/List'
import ListItem from '@material-ui/core/ListItem'
import _ from 'lodash'
import React from 'react'
import { changeSelect, mergeTracks } from '../action/common'
import { changeSelectedLabelsAttributes, deleteSelectedLabels, deleteSelectedTracks, terminateSelectedTracks } from '../action/select'
import { addLabelTag } from '../action/tag'
import { renderButtons, renderTemplate } from '../common/label'
import Session from '../common/session'
import { Key, LabelTypeName } from '../common/types'
import { tracksOverlapping } from '../functional/track'
import { Attribute, State, TrackType } from '../functional/types'
import { Component } from './component'
import { makeButton } from './general_button'
import { Category } from './toolbar_category'

/** This is the interface of props passed to ToolBar */
interface Props {
  /** categories of ToolBar */
  categories: string[] | null
  /** attributes of ToolBar */
  attributes: Attribute[]
  /** itemType of ToolBar 'video' | 'image' */
  itemType: string
  /** labelType of ToolBar 'box2d' | 'polygon2d' | 'lane' */
  labelType: string
}
/**
 * This is ToolBar component that displays
 * all the attributes and categories for the 2D bounding box labeling tool
 */
export class ToolBar extends Component<Props> {
  /** The hashed list of keys currently down */
  private _keyDownMap: { [key: string]: boolean }
  /** key down handler */
  private _keyDownHandler: (e: KeyboardEvent) => void
  /** key up handler */
  private _keyUpHandler: (e: KeyboardEvent) => void
  constructor (props: Readonly<Props>) {
    super(props)
    this.handleToggle = this.handleToggle.bind(this)
    this._keyDownHandler = this.onKeyDown.bind(this)
    this._keyUpHandler = this.onKeyUp.bind(this)
    this.handleAttributeToggle = this.handleAttributeToggle.bind(this)
    this.getAlignmentIndex = this.getAlignmentIndex.bind(this)
    this._keyDownMap = {}
  }

  /**
   * handles keyDown Events
   * @param {keyboardEvent} e
   */
  public onKeyDown (e: KeyboardEvent) {
    const state = this.state
    const select = state.user.select
    switch (e.key) {
      case Key.BACKSPACE:
        if (Object.keys(select.labels).length > 0) {
          const controlDown =
            this.isKeyDown(Key.CONTROL) || this.isKeyDown(Key.META)
          if (controlDown && this.isKeyDown(Key.SHIFT)) {
            // Delete track
            Session.dispatch(deleteSelectedTracks(state))
          } else if (controlDown) {
            // Terminate track
            Session.dispatch(terminateSelectedTracks(state, select.item))
          } else {
            // delete labels
            Session.dispatch(deleteSelectedLabels(state))
          }
        }
        break
      case Key.L_LOW:
      case Key.L_UP:
        // TODO: Move labels up to task level (out of items) and
        // label drawables to Session so that we don't have to search for labels
        if (this.isKeyDown(Key.CONTROL) || this.isKeyDown(Key.META)) {
          this.linkSelectedTracks(state)
        }
        break
    }
    this._keyDownMap[e.key] = true
  }

  /**
   * Key up handler
   * @param e
   */
  public onKeyUp (e: KeyboardEvent) {
    delete this._keyDownMap[e.key]
  }

  /**
   * Add keyDown Event Listener
   */
  public componentDidMount () {
    super.componentDidMount()
    document.addEventListener('keydown', this._keyDownHandler)
    document.addEventListener('keyup', this._keyUpHandler)
  }

  /**
   * Remove keyDown Event Listener
   */
  public componentWillUnmount () {
    super.componentWillUnmount()
    document.removeEventListener('keydown', this._keyDownHandler)
    document.removeEventListener('keyup', this._keyUpHandler)
  }

  /**
   * ToolBar render function
   * @return component
   */
  public render () {
    const { categories, attributes, itemType, labelType } = this.props
    return (
      <div>
        {categories !== null ? (
          <ListItem style={{ textAlign: 'center' }}>
            <Category categories={categories} headerText={'Label Category'} />
          </ListItem>
        ) : null}
        <Divider variant='middle' />
        <List>
          {attributes.map((element: Attribute) =>
            (<React.Fragment key = {element.name}>
            {renderTemplate(
              element.toolType,
              this.handleToggle,
              this.handleAttributeToggle,
              this.getAlignmentIndex,
              element.name,
              element.values
            )}
            </React.Fragment>
            )
          )}
        </List>
        <div>
          <div>{makeButton('Delete', () => {
            Session.dispatch(deleteSelectedLabels(this.state))
          })
          }</div>
          {renderButtons(itemType, labelType)}
        </div>
      </div>
    )
  }

  /**
   * handles tag attribute toggle, dispatching the addLabelTag action
   * @param {string} alignment
   */
  private handleAttributeToggle (toggleName: string, alignment: string) {
    const state = this.state
    const allAttributes = state.task.config.attributes
    const attributeIndex = this.getAttributeIndex(allAttributes, toggleName)
    if (attributeIndex === -1) {
      return
    }
    const currentAttribute = allAttributes[attributeIndex]
    const selectedIndex = currentAttribute.values.indexOf(alignment)
    if (
      state.task.config.labelTypes[state.user.select.labelType] ===
      LabelTypeName.TAG
    ) {
      Session.dispatch(addLabelTag(attributeIndex, selectedIndex))
    } else {
      const currentAttributes = state.user.select.attributes
      const attributes: { [key: number]: number[] } = {}
      for (const keyStr of Object.keys(currentAttributes)) {
        const key = Number(keyStr)
        attributes[key] = currentAttributes[key]
      }
      attributes[attributeIndex] = [selectedIndex]
      Session.dispatch(changeSelectedLabelsAttributes(state, attributes))
      Session.dispatch(changeSelect({ attributes }))
    }
  }
  /**
   * This function updates the checked list of switch buttons.
   * @param {string} switchName
   */
  private handleToggle (switchName: string) {
    const state = this.state
    const allAttributes = state.task.config.attributes
    const toggleIndex = this.getAttributeIndex(allAttributes, switchName)
    if (toggleIndex >= 0) {
      const currentAttributes = state.user.select.attributes
      const attributes: { [key: number]: number[] } = {}
      for (const [key] of allAttributes.entries()) {
        if (currentAttributes[key]) {
          attributes[key] = currentAttributes[key]
        } else {
          attributes[key] = [0]
        }
      }
      if (attributes[toggleIndex][0] > 0) {
        attributes[toggleIndex][0] = 0
      } else {
        attributes[toggleIndex][0] = 1
      }
      if (
          state.task.config.labelTypes[state.user.select.labelType] ===
          LabelTypeName.TAG
        ) {
        Session.dispatch(addLabelTag(toggleIndex, attributes[toggleIndex][0]))
      } else {
        Session.dispatch(changeSelectedLabelsAttributes(state, attributes))
      }
      Session.dispatch(changeSelect({ attributes }))
    }
  }

  /**
   * helper function to get attribute index with respect to the label's
   * attributes
   */
  private getAlignmentIndex (name: string): number {
    const state = this.state
    const attributeIndex = this.getAttributeIndex(
      state.task.config.attributes,
      name
    )
    if (attributeIndex < 0) {
      return 0
    }
    if (
      state.task.config.labelTypes[state.user.select.labelType] ===
      LabelTypeName.TAG
    ) {
      const item = state.task.items[state.user.select.item]
      const labelId = Number(_.findKey(item.labels))
      if (isNaN(labelId)) {
        return 0
      }
      const attributes = item.labels[labelId].attributes
      const index = this.getAttributeIndex(state.task.config.attributes, name)
      if (index < 0) {
        return 0
      }
      if (attributes[index]) {
        return attributes[index][0]
      } else {
        return 0
      }
    } else {
      const currentAttributes = state.user.select.attributes
      return currentAttributes
        ? Object.keys(currentAttributes).indexOf(String(attributeIndex)) >= 0
          ? currentAttributes[attributeIndex][0]
          : 0
        : 0
    }
  }
  /**
   * helper function to get attribute index with respect to the config
   * attributes
   * @param allAttributes
   * @param name
   */
  private getAttributeIndex (allAttributes: Attribute[], toggleName: string
  ): number {
    let attributeIndex = -1
    for (let i = 0; i < allAttributes.length; i++) {
      if (allAttributes[i].name === toggleName) {
        attributeIndex = i
      }
    }
    return attributeIndex
  }

  /**
   * Whether a specific key is pressed down
   * @param {string} key - the key to check
   * @return {boolean}
   */
  private isKeyDown (key: string): boolean {
    return this._keyDownMap[key]
  }

  /**
   * Link selected tracks
   * @param state
   */
  private linkSelectedTracks (state: State) {
    const select = state.user.select
    const tracks: TrackType[] = []
    const trackIds: number[] = []

    for (const key of Object.keys(select.labels)) {
      const index = Number(key)
      for (const labelId of select.labels[index]) {
        const trackId = state.task.items[index].labels[labelId].track
        tracks.push(state.task.tracks[trackId])
      }
    }

    if (!tracksOverlapping(tracks)) {
      Session.dispatch(mergeTracks(trackIds))
    }
  }
}
