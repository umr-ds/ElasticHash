import { SizedModel } from './galleries/AbstractGallery';
export interface RatioLimits {
    min?: number;
    max?: number;
}
export declare function getIcon(document: Document, name: string): SVGSVGElement;
export declare function getImageRatio(model: SizedModel, ratioLimits?: RatioLimits): number;
