import { Column } from '../Column';
import { Item } from '../Item';
import { RatioLimits } from '../Utility';
import { AbstractGallery, GalleryOptions, ModelAttributes } from './AbstractGallery';
export interface MasonryGalleryOptions extends GalleryOptions {
    columnWidth: number;
    ratioLimit?: RatioLimits;
}
export declare class Masonry<Model extends ModelAttributes = ModelAttributes> extends AbstractGallery<Model> {
    /**
     * Options after having been defaulted
     */
    protected options: Required<MasonryGalleryOptions>;
    /**
     * Regroup the list of columns
     */
    protected columns: Column<Model>[];
    constructor(elementRef: HTMLElement, options: MasonryGalleryOptions, photoswipeElementRef?: HTMLElement | null, scrollElementRef?: HTMLElement | null);
    /**
     * Compute sides with 1:1 ratio
     */
    static organizeItems<T extends ModelAttributes>(gallery: Masonry<T>, items: Item<T>[], fromIndex?: number, toIndex?: number | null): void;
    init(): void;
    organizeItems(items: Item<Model>[], fromRow?: number, toRow?: number): void;
    protected initItems(): void;
    protected onScroll(): void;
    protected onPageAdd(): void;
    protected getEstimatedColumnsPerRow(): number;
    protected getEstimatedRowsPerPage(): number;
    /**
     * Use current gallery height as reference. To fill free space it add images until the gallery height changes, then are one more row
     */
    protected addUntilFill(): void;
    protected addItemToDOM(item: Item<Model>): void;
    protected endResize(): void;
    protected addColumns(): void;
    protected empty(): void;
    /**
     * Returns true if at least one columns doesn't overflow on the bottom of the viewport
     */
    private viewPortIsNotFilled;
    private addItemsToDom;
    /**
     * Return square side size
     */
    private getColumnWidth;
    private getShortestColumn;
}
