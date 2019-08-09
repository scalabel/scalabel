import * as fa from '@fortawesome/free-solid-svg-icons/index'
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome'
import { IconButton } from '@material-ui/core'
import Grid from '@material-ui/core/Grid'
import Input from '@material-ui/core/Input'
import Slider from '@material-ui/core/Slider'
import { withStyles } from '@material-ui/core/styles/index'
import _ from 'lodash'
import React, { ChangeEvent } from 'react'
import { goToItem } from '../action/common'
import Session from '../common/session'
import { playerControlStyles } from '../styles/label'
import { Component } from './component'

interface ClassType {
  /** play button */
  button: string
  /** player control */
  playerControl: string
  /** input */
  input: string
  /** underline */
  underline: string
  /** slider */
  slider: string
}

interface Props {
  /** Styles of TitleBar */
  classes: ClassType
  /** Number of frames */
  num_frames: number
}

/**
 * Go to the next Item
 * @param {number} index
 */
function goToItemWithIndex (index: number) {
  Session.dispatch(goToItem(index - 1))
}

/**
 * Player control
 */
class PlayerControl extends Component<Props> {

  /** if the slider is playing */
  private playing: boolean
  /** the current frame */
  private currentFrame: number
  /** interval function */
  private intervalId: number
  /** number of frames */
  private numFrames: number

  public constructor (props: Readonly<Props>) {
    super(props)
    this.playing = false
    this.currentFrame = 1
    this.intervalId = -1
    const { num_frames } = this.props
    this.numFrames = num_frames
  }

  /**
   * Render function
   * @return {React.Fragment} React fragment
   */
  public render () {
    const { classes } = this.props
    let playIcon
    if (this.playing) {
      playIcon = fa.faPause
    } else {
      playIcon = fa.faPlay
    }
    return (
      <div className={classes.playerControl}>
        <Grid container alignItems='center'
          spacing={1}>
          <Grid item>
            <IconButton className={classes.button}
              onClick={() => {
                this.currentFrame = Math.max(this.currentFrame - 1, 1)
                goToItemWithIndex(this.currentFrame)
              }}>
              <FontAwesomeIcon icon={fa.faAngleLeft} size='lg' />
            </IconButton>
            <IconButton className={classes.button}
              onClick={() => {
                this.currentFrame = Math.min(
                  this.currentFrame + 1, this.numFrames)
                goToItemWithIndex(this.currentFrame)
              }}>
              <FontAwesomeIcon icon={fa.faAngleRight} size='lg' />
            </IconButton>
            <Input
              className={classes.input}
              value={this.currentFrame}
              margin='dense'
              onChange={(e) => this.handleInputChange(e)}
              inputProps={{
                'step': 1,
                'min': 1,
                'max': this.numFrames,
                'type': 'number',
                'aria-labelledby': 'input-slider'
              }}
            />
            <IconButton className={classes.button}
              onClick={() => this.togglePlay()}>
              <FontAwesomeIcon icon={playIcon} size='xs' />
            </IconButton>
          </Grid>
          <Grid item xs>
            <Slider
              className={classes.slider}
              value={this.currentFrame}
              onChange={(e, newVal) =>
                this.handleSliderChange(e, newVal)}
              aria-labelledby='input-slider'
              min={1}
              max={this.numFrames}
            />
          </Grid>
        </Grid>
      </div>
    )
  }

  /**
   * Handler on slider change
   * @param {ChangeEvent<{}>} _event
   * @param {number | number[]} newValue
   */
  private handleSliderChange (_event: ChangeEvent<{}>,
                              newValue: number | number[]) {
    if (Array.isArray(newValue)) {
      this.currentFrame = newValue[0]
    } else {
      this.currentFrame = newValue
    }
    goToItemWithIndex(this.currentFrame)
  }

  /**
   * Handler on slider change
   * @param {React.ChangeEvent<HTMLInputElement | HTMLTextAreaElement>} event
   */
  private handleInputChange (
    event: React.ChangeEvent<HTMLInputElement | HTMLTextAreaElement>) {
    const value = Number(event.target.value)
    this.currentFrame = Math.max(Math.min(value, this.numFrames), 1)
    goToItemWithIndex(this.currentFrame)
  }

  /**
   * Callback function for clicking play/pause
   */
  private togglePlay () {

    // switch the play status
    this.playing = !this.playing

    // update the icon and play/pause the vid
    if (this.playing) {
      this.intervalId = window.setInterval(() => {
        if (this.currentFrame < this.numFrames) {
          this.currentFrame += 1
          goToItemWithIndex(this.currentFrame)
        } else {
          this.togglePlay()
        }
      },
        100)
    } else {
      window.clearInterval(this.intervalId)
    }
    this.forceUpdate()
  }
}

export default withStyles(
  playerControlStyles, { withTheme: true })(PlayerControl)
