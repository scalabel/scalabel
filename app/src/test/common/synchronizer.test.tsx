import { cleanup } from '@testing-library/react'
import io from 'socket.io-client'
import uuid4 from 'uuid/v4'
import { setStatusToUnsaved } from '../../js/action/common'
import { AddLabelsAction } from '../../js/action/types'
import Session from '../../js/common/session'
import { Synchronizer } from '../../js/common/synchronizer'
import { index2str } from '../../js/common/util'
import * as selector from '../../js/functional/selector'
import {
  ActionPacketType, EventName, RegisterMessageType,
  SyncActionMessageType } from '../../js/server/types'
import { updateState } from '../../js/server/util'
import { getInitialState, getRandomBox2dAction } from '../server/util/util'

let sessionId: string
let botSessionId: string
let taskIndex: number
let projectName: string
let userId: string
let bots: boolean
let autosave: boolean
const socketEmit = jest.fn()
const mockSocket = {
  on: jest.fn(),
  connected: true,
  emit: socketEmit
}

beforeAll(() => {
  sessionId = 'fakeSessId'
  botSessionId = 'botSessId'
  taskIndex = 0
  projectName = 'testProject'
  userId = 'fakeUserId'
  bots = false
  autosave = true
})

beforeEach(() => {
  Session.dispatch(setStatusToUnsaved())
})

afterEach(cleanup)
describe('Test synchronizer functionality', () => {
  test('Test correct registration message gets sent', async () => {
    // Since this deals with registration, don't initialize the state
    const initializeState = false
    const sync = startSynchronizer(initializeState)
    sync.sendConnectionMessage('')

    // Frontend doesn't have a session id until after registration
    const expectedSessId = ''
    checkConnectMessage(expectedSessId)
    expect(selector.isStatusUnsaved(Session.store.getState())).toBe(true)
  })

  test('Test send-ack loop', async () => {
    const sync = startSynchronizer()
    dispatchAndCheckActions(sync, 1)

    // After ack arrives, no actions are queued anymore
    const ackPackets = sendAcks(sync)
    expect(sync.numQueuedActions).toBe(0)
    expect(sync.numLoggedActions).toBe(1)
    expect(selector.isStatusSaved(Session.store.getState())).toBe(true)

    // If second ack arrives, it is ignored
    sync.handleBroadcast(
      packetToMessage(ackPackets[0]))
    expect(sync.numLoggedActions).toBe(1)
  })

  test('Test model prediction status', async () => {
    const sync = startSynchronizer()
    Session.bots = true

    dispatchAndCheckActions(sync, 2)

    // After acks arrive, session status is marked as computing
    const ackPackets = sendAcks(sync)
    expect(selector.isStatusComputing(Session.store.getState())).toBe(true)

    // Mark computation as finished when all model actions arrive
    const modelPackets = []
    for (const ackPacket of ackPackets) {
      const modelAction = getRandomBox2dAction()
      modelAction.sessionId = botSessionId
      const modelPacket: ActionPacketType = {
        actions: [modelAction],
        id: uuid4(),
        triggerId: ackPacket.id
      }

      modelPackets.push(modelPacket)
    }

    sync.handleBroadcast(
      packetToMessageBot(modelPackets[0]))
    expect(sync.numActionsPendingPrediction).toBe(1)
    expect(selector.isComputeDone(Session.store.getState())).toBe(false)

    sync.handleBroadcast(
      packetToMessageBot(modelPackets[1]))
    expect(sync.numActionsPendingPrediction).toBe(0)
    expect(selector.isComputeDone(Session.store.getState())).toBe(true)
  })

  test('Test reconnection', async () => {
    const sync = startSynchronizer()
    const frontendActions = dispatchAndCheckActions(sync, 1)

    // Backend disconnects instead of acking
    sync.handleDisconnect()
    expect(selector.isStatusReconnecting(Session.store.getState())).toBe(true)

    // Reconnect, but some missed actions occured in the backend
    const newInitialState = updateState(
      getInitialState(sessionId),
      [getRandomBox2dAction()]
    )
    sync.sendConnectionMessage(sessionId)
    checkConnectMessage(sessionId)
    sync.finishRegistration(newInitialState, autosave, sessionId, bots)

    /**
     * Check that frontend state updates correctly
     * Except for session state, which will change because of status effects
     */
    const expectedState = updateState(newInitialState, frontendActions)
    expect(Session.getState().task).toMatchObject(expectedState.task)
    expect(Session.getState().user).toMatchObject(expectedState.user)
    // Also check that save is still in progress
    expect(sync.numQueuedActions).toBe(1)
    checkActionsAreQueued(sync, frontendActions)
    expect(selector.isStatusSaving(Session.store.getState())).toBe(true)

    // After ack arrives, no actions are queued anymore
    sendAcks(sync)
    expect(Session.getState().task).toMatchObject(expectedState.task)
    expect(Session.getState().user).toMatchObject(expectedState.user)
    expect(sync.numQueuedActions).toBe(0)
    expect(sync.numLoggedActions).toBe(1)
  })
})

/**
 * Dispatch and check the effects of a single add label action
 */
function dispatchAndCheckActions (
  sync: Synchronizer, numActions: number): AddLabelsAction[] {
  // Dispatch actions to trigger sync events
  const actions: AddLabelsAction[] = []
  for (let _ = 0; _ < numActions; _++) {
    const action = getRandomBox2dAction()
    sync.logAction(action, autosave, sessionId, bots)
    actions.push(action)
  }

  // Verify the synchronizer state before ack arrives
  expect(sync.numQueuedActions).toBe(numActions)
  checkActionsAreQueued(sync, actions)
  expect(selector.isStatusSaving(Session.store.getState())).toBe(true)
  if (Session.bots) {
    expect(sync.numActionsPendingPrediction).toBe(numActions)
  }
  return actions
}

/**
 * Acks all the waiting packets
 */
function sendAcks (sync: Synchronizer): ActionPacketType[] {
  const actionPackets = sync.listActionPackets()
  for (const actionPacket of actionPackets) {
    sync.handleBroadcast(
      packetToMessage(actionPacket))
  }
  return actionPackets
}

/**
 * Check that the action has been be queued
 */
function checkActionsAreQueued (
  sync: Synchronizer, actions: AddLabelsAction[]) {
  const actionPackets = sync.listActionPackets()
  expect(actionPackets.length).toBe(actions.length)
  for (let i = 0; i < actions.length; i++) {
    expect(actionPackets[i].actions[0]).toBe(actions[i])
  }
}

/**
 * Checkthat correct connection message was sent
 */
function checkConnectMessage (sessId: string) {
  const expectedMessage: RegisterMessageType = {
    projectName,
    taskIndex,
    sessionId: sessId,
    userId,
    address: location.origin,
    bot: false
  }
  expect(socketEmit).toHaveBeenCalledWith(EventName.REGISTER, expectedMessage)
}

/**
 * Start the browser synchronizer being tested
 */
function startSynchronizer (setInitialState: boolean = true): Synchronizer {
  io.connect = jest.fn().mockImplementation(() => mockSocket)
  const synchronizer = new Synchronizer(
    mockSocket as any,
    taskIndex,
    projectName,
    userId
  )

  if (setInitialState) {
    const initialState = getInitialState(sessionId)
    synchronizer.finishRegistration(initialState,
      initialState.task.config.autosave,
      initialState.session.id,
      initialState.task.config.bots)
  }

  // Initially, no actions are queued for saving
  expect(synchronizer.numQueuedActions).toBe(0)
  expect(selector.isStatusUnsaved(Session.store.getState())).toBe(true)

  return synchronizer
}

/**
 * Convert action packet to sync message
 */
function packetToMessage (packet: ActionPacketType): SyncActionMessageType {
  return {
    actions: packet,
    projectName,
    sessionId,
    taskId: index2str(taskIndex),
    bot: false
  }
}

/**
 * Convert action packet to sync message from a bot
 */
function packetToMessageBot (packet: ActionPacketType): SyncActionMessageType {
  return {
    actions: packet,
    projectName,
    sessionId: botSessionId,
    taskId: index2str(taskIndex),
    bot: true
  }
}
