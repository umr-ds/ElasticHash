import { Item } from '../Item';
import { RatioLimits } from '../Utility';
import { GalleryOptions, ModelAttributes, SizedModel } from './AbstractGallery';
import { AbstractRowGallery } from './AbstractRowGallery';
export interface NaturalGalleryOptions extends GalleryOptions {
    rowHeight: number;
    ratioLimit?: RatioLimits;
}
export declare class Natural<Model extends ModelAttributes = ModelAttributes> extends AbstractRowGallery<Model> {
    /**
     * Options after having been defaulted
     */
    protected options: NaturalGalleryOptions & Required<GalleryOptions>;
    constructor(elementRef: HTMLElement, options: NaturalGalleryOptions, photoswipeElementRef?: HTMLElement | null, scrollElementRef?: HTMLElement | null);
    static organizeItems<T extends ModelAttributes>(gallery: Natural<T>, items: Item<T>[], fromRow?: number, toRow?: number | null, currentRow?: number | null): void;
    /**
     * Compute sizes for given images to fit in given row width
     * Items are updated
     */
    static computeSizes<T extends ModelAttributes>(chunk: Item<T>[], containerWidth: number | null, margin: number, row: number, maxRowHeight?: number | null, ratioLimits?: RatioLimits): void;
    static getRowWidth(models: SizedModel[], maxRowHeight: number, margin: number, ratioLimits?: RatioLimits): number;
    static getRowHeight(models: SizedModel[], containerWidth: number, margin: number, ratioLimits?: RatioLimits): number;
    /**
     * Return the ratio format of models as if they where a single image
     */
    static getRatios(models: SizedModel[], ratioLimits?: RatioLimits): number;
    addRows(rows: number): void;
    organizeItems(items: Item<Model>[], fromRow?: number, toRow?: number): void;
    protected endResize(): void;
    protected getEstimatedColumnsPerRow(): number;
    protected getEstimatedRowsPerPage(): number;
    private completeLastRow;
}
