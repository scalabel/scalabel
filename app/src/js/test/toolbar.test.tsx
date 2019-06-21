import * as React from 'react';
import { create } from 'react-test-renderer';
import {ListItemText} from '@material-ui/core';
import {Category} from '../components/toolbar_category';
import FormControlLabel from '@material-ui/core/FormControlLabel/FormControlLabel';
import { cleanup, render, fireEvent } from '@testing-library/react';

afterEach(cleanup);

describe('Toolbar category setting', () => {
    test('Category selection', () => {
        const {getByLabelText} = render(
            <Category categories={['A', 'B']}/>);
        const selectedValued = getByLabelText(/A/i);
        expect(selectedValued.getAttribute('value')).toEqual('A');
        const radio = getByLabelText('A');
        fireEvent.change(radio, {target: {value: 'B'}});
        // expect state to be changed
        expect(radio.getAttribute('value')).toBe('B');
    });

    test('Test elements in Category', () => {
        const category = create(
            <Category categories={['A', 'B']} />);
        const root = category.root;
        expect(root.props.categories[0].toString()).toBe('A');
        expect(root.props.categories[1].toString()).toBe('B');
        expect(root.findByType(ListItemText).props.primary)
            .toBe('Label Category');
    });

    test('Category by type', () => {
        const category = create(
            <Category categories={['OnlyCategory']} />);
        const root = category.root;
        expect(root.findByType(FormControlLabel).props.label)
            .toBe('OnlyCategory');
    });

    test('Null category', () => {
        const category = create(<Category categories={null} />);
        const root = category.getInstance();
        expect(root).toBe(null);
    });
});
