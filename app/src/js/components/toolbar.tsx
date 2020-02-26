import Divider from '@material-ui/core/Divider'
import ListItem from '@material-ui/core/ListItem'
import _ from 'lodash'
import React from 'react'
import { changeViewerConfig, mergeTracks } from '../action/common'
import { deleteSelectedLabels, deleteSelectedTracks, terminateSelectedTracks } from '../action/select'
import Session from '../common/session'
import { Key } from '../common/types'
import { tracksOverlapping } from '../functional/track'
import { Attribute, State, TrackType } from '../functional/types'
import { Component } from './component'
import { makeButton } from './general_button'
import { Category } from './toolbar_category'
import { AttributeSelector } from './attribute_selector'

/** This is the interface of props passed to ToolBar */
interface Props {
  /** categories of ToolBar */
  categories: string[] | null
  /** attributes of ToolBar */
  attributes: Attribute[]
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
    this._keyDownHandler = this.onKeyDown.bind(this)
    this._keyUpHandler = this.onKeyUp.bind(this)
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
      case Key.H_LOW:
      case Key.H_UP:
        const config = {
          ...this.state.user.viewerConfigs[Session.activeViewerId]
        }
        config.hideLabels = !config.hideLabels
        Session.dispatch(changeViewerConfig(Session.activeViewerId, config))
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
    const { categories } = this.props
    return (
      <div>
        {categories !== null ? (
          <ListItem style={{ textAlign: 'center' }}>
            <Category categories={categories} headerText={'Label Category'} />
          </ListItem>
        ) : null}
        <Divider variant='middle' />
        <AttributeSelector/>
        <div>
          <div>{makeButton('Delete', () => {
            Session.dispatch(deleteSelectedLabels(this.state))
          })
          }</div>
        </div>
      </div>
    )
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
