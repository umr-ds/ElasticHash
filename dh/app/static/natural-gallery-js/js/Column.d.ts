import { Item } from './Item';
import { ModelAttributes } from './galleries/AbstractGallery';
export declare interface ColumnOptions {
    width: number;
    gap: number;
}
export declare class Column<Model extends ModelAttributes> {
    private options;
    private readonly collection;
    private readonly _elementRef;
    constructor(document: Document, options: ColumnOptions);
    addItem(item: Item<Model>): void;
    get height(): number;
    get length(): number;
    get elementRef(): HTMLElement;
}
