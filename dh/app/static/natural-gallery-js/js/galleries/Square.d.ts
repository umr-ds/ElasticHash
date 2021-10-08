import { Item } from '../Item';
import { GalleryOptions, ModelAttributes } from './AbstractGallery';
import { AbstractRowGallery } from './AbstractRowGallery';
export interface SquareGalleryOptions extends GalleryOptions {
    itemsPerRow: number;
}
export declare class Square<Model extends ModelAttributes = ModelAttributes> extends AbstractRowGallery<Model> {
    /**
     * Options after having been defaulted
     */
    protected options: SquareGalleryOptions & Required<GalleryOptions>;
    constructor(elementRef: HTMLElement, options: SquareGalleryOptions, photoswipeElementRef?: HTMLElement | null, scrollElementRef?: HTMLElement | null);
    /**
     * Compute sides with 1:1 ratio
     */
    static organizeItems<T extends ModelAttributes>(gallery: Square<T>, items: Item<T>[], firstRowIndex?: number, toRow?: number | null): void;
    protected getEstimatedColumnsPerRow(): number;
    protected getEstimatedRowsPerPage(): number;
    /**
     * Return square side size
     */
    protected getItemSideSize(): number;
    organizeItems(items: Item<Model>[], fromRow?: number, toRow?: number): void;
}
