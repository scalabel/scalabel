import ListItem from '@material-ui/core/ListItem/ListItem';
import {SwitchBtn} from '../components/toolbar_switch';
import {genButton} from '../components/general_button';
import {ListButton} from '../components/toolbar_listButton';
import React from 'react';

export function renderTemplate(toolType: any, handleToggle: any, name: any, values: any) {
    if (toolType === 'switch') {
        return (
            <SwitchBtn onChange = {handleToggle} value = {name} />
    );
    } else if (toolType === 'list') {
        return (
            <ListItem style={{textAlign: 'center'}} >
                <ListButton name = {name} values = {values} />
            </ListItem>
        );
    }
}

export function renderButtons(itemType: any, labelType: any) {
    if (itemType === 'video') {
        return (
            <div>
                <div>
                    {genButton({name: 'End Object Trac'})}
                </div>
                <div>
                    {genButton({name: 'Track-Link'})}
                </div>
            </div>
        );
    }
    if (labelType === 'box2d') {
        // do nothing
    } else if (labelType === 'segmentation' || labelType === 'lane') {
        if (labelType === 'segmentation') {
            if (itemType === 'image') {
                return (
                    <div>
                        <div>
                            {genButton({name: 'Link'})}
                        </div>
                        <div>
                            {genButton({name: 'Quick-draw'})}
                         </div>
                    </div>
                );
            }
        } else if (labelType === 'lane') {
            // do nothing
        }
    }
}
