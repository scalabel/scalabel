import _ from 'lodash'
import OrderedMap from 'orderedmap'
import uuid4 from 'uuid/v4'
import {
  setStatusAfterConnect,
  setStatusToComputeDone,
  setStatusToComputing,
  setStatusToReconnecting,
  setStatusToSaved,
  setStatusToSaving,
  setStatusToSubmitted,
  setStatusToSubmitting,
  setStatusToUnsaved,
  updateTask} from '../action/common'
import * as types from '../action/types'
import { isSessionFullySaved } from '../functional/selector'
import { State } from '../functional/types'
import { SocketClient } from '../server/socket_interface'
import { ActionPacketType, EventName, RegisterMessageType,
  SyncActionMessageType } from '../server/types'
import Session from './session'
import { setupSession } from './session_setup'
import { index2str } from './util'

const CONFIRMATION_MESSAGE =
  'You have unsaved changes that will be lost if you leave this page. '

/**
 * Synchronizes data with other sessions
 */
export class Synchronizer {

  /**
   * Getter for number of logged (acked) actions
   */
  public get numLoggedActions (): number {
    return this.actionLog.length
  }

  /**
   * Get number of actions in the process of being saved
   */
  public get numActionsPendingSave (): number {
    return this.actionsPendingSave.size
  }

  /**
   * Getter for number of actions with predictions running
   */
  public get numActionsPendingPrediction (): number {
    return this.actionsPendingPrediction.size
  }

  /** Socket connection */
  public socket: SocketClient
  /** Name of the project */
  public projectName: string
  /** Index of the task */
  public taskIndex: number
  /** The user/browser id, constant across sessions */
  public userId: string
  /** Actions queued to be sent to the backend */
  public actionQueue: types.BaseAction[]
  /**
   * Actions in the process of being saved, mapped by packet id
   * OrderedMap ensures that resending keeps the same order
   */
  private actionsPendingSave: OrderedMap<ActionPacketType>
  /** Timestamped log for completed actions */
  private actionLog: types.BaseAction[]
  /** Log of packets that have been acked */
  private ackedPackets: Set<string>
  /** The ids of action packets pending model predictions */
  private actionsPendingPrediction: Set<string>
  /** Flag for initial registration completion */
  private registeredOnce: boolean
  /** Name of the DOM container */
  private containerName: string

  constructor (
    socket: SocketClient,
    taskIndex: number, projectName: string,
    userId: string, containerName: string= '') {
    this.socket = socket
    this.taskIndex = taskIndex
    this.projectName = projectName
    this.containerName = containerName

    this.actionQueue = []
    this.actionsPendingSave = OrderedMap.from()
    this.actionLog = []
    this.userId = userId
    this.ackedPackets = new Set()
    this.actionsPendingPrediction = new Set()
    this.registeredOnce = false

    window.onbeforeunload = this.warningPopup.bind(this)
  }

  /**
   * Queue a new action for saving
   */
  public queueActionForSaving (action: types.BaseAction, autosave: boolean,
                               sessionId: string, bots: boolean) {
    // Exclude actions from other sessions and actions on non-shared state
    if (sessionId === action.sessionId && !action.frontendOnly &&
      !types.isSessionAction(action)) {
      this.actionQueue.push(action)
      if (autosave) {
        this.save(sessionId, bots)
      } else {
        Session.dispatch(setStatusToUnsaved())
      }
    }
  }

  /**
   * Displays pop-up warning user when leaving with unsaved changes
   */
  public warningPopup (e: BeforeUnloadEvent) {
    if (!isSessionFullySaved(Session.store.getState())) {
      e.returnValue = CONFIRMATION_MESSAGE // Gecko + IE
      return CONFIRMATION_MESSAGE // Gecko + Webkit, Safari, Chrome etc.
    }
  }

  /**
   * Registers the session with the backend, triggering a register ack
   */
  public sendConnectionMessage (sessionId: string) {
    const message: RegisterMessageType = {
      projectName: this.projectName,
      taskIndex: this.taskIndex,
      sessionId,
      userId: this.userId,
      address: location.origin,
      bot: false
    }
    /* Send the registration message to the backend */
    this.socket.emit(EventName.REGISTER, message)
    Session.dispatch(setStatusAfterConnect())
  }

  /**
   * Initialized synced state, and sends any queued actions
   */
  public finishRegistration (
    state: State, autosave: boolean, sessionId: string, bots: boolean) {
    if (!this.registeredOnce) {
      // One-time setup after first registration
      this.registeredOnce = true
      setupSession(state, this.containerName)
    } else {
      // Get the local session in-sync after a disconnect/reconnect
      if (autosave) {
        // Update with any backend changes that occurred during disconnect
        Session.dispatch(updateTask(state.task))

        // Re-apply frontend task actions after updating task from backend
        for (const actionPacket of this.listActionsPendingSave()) {
          for (const action of actionPacket.actions) {
            if (types.isTaskAction(action)) {
              action.frontendOnly = true
              Session.dispatch(action)
            }
          }
        }
      }

      for (const actionPacket of this.listActionsPendingSave()) {
        this.sendActions(actionPacket, sessionId)
      }
      if (autosave) {
        this.save(sessionId, bots)
      }
    }
  }

  /**
   * Called when backend sends ack for actions that were sent to be synced
   * Updates relevant queues and syncs actions from other sessions
   */
  public handleBroadcast (message: SyncActionMessageType) {
    const actionPacket = message.actions
    // Remove stored actions when they are acked
    this.actionsPendingSave = this.actionsPendingSave.remove(actionPacket.id)

    // If action was already acked, ignore it
    if (this.ackedPackets.has(actionPacket.id)) {
      return
    }
    this.ackedPackets.add(actionPacket.id)

    for (const action of actionPacket.actions) {
      // actionLog matches backend action ordering
      this.actionLog.push(action)
      if (action.sessionId !== Session.id) {
        if (types.isTaskAction(action)) {
          // Dispatch any task actions broadcasted from other sessions
          Session.dispatch(action)
        }
      }
    }

    if (this.actionsPendingPrediction.has(actionPacket.id)) {
      /* Original action was acked by the server
       * This means the bot also received the action
       * And started its prediction */
      Session.dispatch(setStatusToComputing())
    } else if (actionPacket.triggerId !== undefined &&
      this.actionsPendingPrediction.has(actionPacket.triggerId)) {
      // Ack of bot action means prediction is finished
      this.actionsPendingPrediction.delete(actionPacket.triggerId)
      if (this.actionsPendingPrediction.size === 0) {
        Session.dispatch(setStatusToComputeDone())
      }
    } else if (message.sessionId === Session.id) {
      if (types.hasSubmitAction(actionPacket.actions)) {
        Session.dispatch(setStatusToSubmitted())
      } else if (this.actionsPendingSave.size === 0) {
        // Once all actions being saved are acked, update the save status
        Session.dispatch(setStatusToSaved())
      }
    }
  }

  /**
   * Called when session disconnects from backend
   */
  public handleDisconnect () {
    Session.dispatch(setStatusToReconnecting())
  }

  /**
   * Converts ordered map of action packets to a list
   * Order of the list should match the order in which keys were added
   */
  public listActionsPendingSave (): ActionPacketType[] {
    const values: ActionPacketType[] = []
    if (this.actionsPendingSave.size > 0) {
      this.actionsPendingSave.forEach(
        (_key: string, value: ActionPacketType) => {
          values.push(value)
        })
    }
    return values
  }

  /**
   * Send all queued actions to the backend
   * and move actions to actionsPendingSave
   */
  public save (sessionId: string, bots: boolean) {
    if (this.socket.connected) {
      if (this.actionQueue.length > 0) {
        const packet: ActionPacketType = {
          actions: this.actionQueue,
          id: uuid4()
        }
        this.actionsPendingSave =
          this.actionsPendingSave.update(packet.id, packet)
        if (this.doesPacketTriggerModel(packet, bots)) {
          this.actionsPendingPrediction.add(packet.id)
        }
        this.sendActions(packet, sessionId)
        this.actionQueue = []
      }
    }
  }

  /**
   * Checks if the action packet contains
   * any actions that would trigger a model query
   */
  public doesPacketTriggerModel (
    actionPacket: ActionPacketType, bots: boolean): boolean {
    if (!bots) {
      return false
    }
    for (const action of actionPacket.actions) {
      if (action.type === types.ADD_LABELS) {
        return true
      }
    }
    return false
  }

  /**
   * Send given action packet to the backend
   * Can be called multiple times if previous attempts aren't acked
   */
  public sendActions (
    actionPacket: ActionPacketType, sessionId: string) {
    const message: SyncActionMessageType = {
      taskId: index2str(this.taskIndex),
      projectName: this.projectName,
      sessionId,
      actions: actionPacket,
      bot: false
    }
    this.socket.emit(EventName.ACTION_SEND, message)
    if (types.hasSubmitAction(actionPacket.actions)) {
      Session.dispatch(setStatusToSubmitting())
    } else {
      Session.dispatch(setStatusToSaving())
    }
  }
}

export default Synchronizer
