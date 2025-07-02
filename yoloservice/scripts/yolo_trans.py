#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  :yolo_trans.py
# @Time      :2025/6/25 09:20:53
# @Author    :雨霓同学
# @Project   :BTD
# @Function  :实现数据集的转换,分割，配置文件生成
import argparse
import sys
import yaml
import shutil
import logging
from pathlib import Path

current_path = Path(__file__).parent.parent.resolve()
utils_path = current_path / 'utils'
if str(current_path) not in sys.path:
    sys.path.insert(0, str(current_path))
if str(utils_path) not in sys.path:
    sys.path.insert(1, str(utils_path))

from sklearn.model_selection import train_test_split  #  pip install scikit-learn

from utils.performance_utils import time_it
from utils.paths import (YOLOSERVER_ROOT,
                RAW_IMAGES_DIR,
                ORIGINAL_ANNOTATIONS_DIR,
                YOLO_STAGED_LABELS_DIR,
                DATA_DIR,
                CONFIGS_DIR,
                LOGS_DIR
                )
from utils.logging_utils import setup_logger
from utils.data_utils import convert_annotations_to_yolo

logger = setup_logger(
    base_path=LOGS_DIR,
    log_type="yolo_trans",
    model_name=None,
    log_level=logging.INFO,
    logger_name="YOLO Trans"
)

class YOLODatasetProcessor:
    """
    一个集成类，负责
    1. 协调原始标注到YOLO TXT 格式的转换
    2. 划分原始图像和转换、复制后的YOLO TXT标签为训练集，验证集，测试集
    3. 生成data.yaml配置文件
    """
    def __init__(self,train_rate=0.8,valid_rate=0.1,annotation_format="coco",
                coco_task="detection",
                final_classes_order=None,
                coco_cls91to80=False):
        """
        初始化数据集处理器
        :param train_rate: 训练集的比例：默认0.8
        :param valid_rate: 验证集的比例：默认0.1
        """
        self.project_root_path = YOLOSERVER_ROOT
        self.raw_images_path = RAW_IMAGES_DIR
        self.yolo_staged_labels_dir = YOLO_STAGED_LABELS_DIR
        self.output_data_path = DATA_DIR
        self.config_path = CONFIGS_DIR
        self.classes = [] if final_classes_order is None else final_classes_order
        self.train_rate = train_rate
        self.valid_rate = valid_rate
        self.test_rate = 1 - train_rate - valid_rate
        self.annotation_format = annotation_format
        self.coco_task = coco_task
        self.coco_cls91to80 = coco_cls91to80


        # 确保数据集比例划分有效,这属于核心业务逻辑验证
        if not (0.0 <= self.train_rate <= 1.0 and
                0.0 <= self.valid_rate <= 1.0 and
                0.0 <= self.test_rate <= 1.0 and
                abs(self.train_rate + self.valid_rate + self.test_rate - 1.0) <= 1e-6):
            logger.error("训练集比例、验证集比例和测试集比例之和必须等于1.0,当前配置无效，请检查配置")
            raise ValueError("数据集比例配置无效/错误")

        self.config_path.mkdir(parents=True, exist_ok=True)
        self.output_dirs = {
            "train": {"images": self.output_data_path / "train" / "images",
                    "labels": self.output_data_path / "train" / "labels"},
            "val": {"images": self.output_data_path / "val" / "images",
                    "labels": self.output_data_path / "val" / "labels"},
            "test": {"images": self.output_data_path / "test" / "images",
                    "labels": self.output_data_path / "test" / "labels"}
                    }
    # 检查原始图像文件，以及转换之后的标注文件是否存在
    def _check_staged_data_existence(self):
        """
        检查转换后的数据集是否存在
        :return: True: 存在，False: 不存在
        """
        # 确保YOLO_STAGED_LABELS_DIR目录存在,且不为空，因为它直接影响后续的分割
        if not self.yolo_staged_labels_dir.exists() or not any(self.yolo_staged_labels_dir.glob("*.txt")):
            logger.error(f"转换后的YOLO TXT 文件目录{self.yolo_staged_labels_dir} "
                        f"中不存在 YOLO TXT 文件，请检查转换是否成功")
            raise FileNotFoundError(f"转换后的YOLO TXT 文件目录{self.yolo_staged_labels_dir} "
                        f"中不存在 YOLO TXT 文件，请检查转换是否成功")
        if not self.raw_images_path.exists() or not any(self.raw_images_path.glob("*.jpg")):
            logger.error(f"原始图像目录{self.raw_images_path} 中不存在图片文件，请检查原始数据集是否正确")
            raise FileNotFoundError(f"原始图像目录{self.raw_images_path} 中不存在图片文件，请检查原始数据集是否正确")
        logger.info(f"原始图像及标注文件暂存区通过检查："
                    f"图像位于 '{self.raw_images_path.relative_to(self.project_root_path)}'"
                    f"标签位于 '{self.yolo_staged_labels_dir.relative_to(self.project_root_path)}'")

    # 确保最终划分后的训练集，测试集，验证集的目录存在以及它们的子目录也存在
    def _ensure_output_dirs_exist(self):
        """
        确保最终划分后的训练集，测试集，验证集的目录存在以及它们的子目录也存在
        :return:
        """
        for split_info in self.output_dirs.values():
            for dir_path in split_info.values():
                dir_path.mkdir(parents=True, exist_ok=True)
                logger.info(f"已创建目录或确认目录存在 '{dir_path.relative_to(self.project_root_path)}'")
        logger.info(f"所有输出数据集分割目录已准备就绪".center(60, "="))

    def _find_matching_files(self):
        """
        找到匹配的图像文件和对应的YOLO TXT文件
        :return: 匹配的图像文件的路径
        """
        txt_files = list(self.yolo_staged_labels_dir.glob("*.txt"))
        if not txt_files:
            logger.warning(f"未找到匹配的YOLO TXT文件，请检查转换是否成功")
            return []

        matching_pairs = []
        img_extensions = [".jpg", ".jpeg", ".png", "bmp", ".tiff", ".webp"]

        for txt_file in txt_files:
            found_image = False
            for ext in img_extensions:
                img_name_stem = txt_file.stem
                image_path = self.raw_images_path / (img_name_stem + ext)
                if image_path.exists():
                    matching_pairs.append((image_path, txt_file))
                    found_image = True
                    break
            if not found_image:
                logger.warning(f"未在 '{self.raw_images_path.relative_to(self.project_root_path)}' "
                            f"中找到匹配的图像文件: '{txt_file.name}'，跳过该标签文件")
        if not matching_pairs:
            logger.error(f"未找到匹配的图像文件，请检查原始数据集是否正确")
        else:
            logger.info(f"找到匹配的图像文件，共 {len(matching_pairs)} 个")
        return matching_pairs

    # 将数据集进行分割，分为训练集，测试集，验证集
    def _split_and_process_data(self,matching_pairs):
        """
        将数据集进行分割，分为训练集，测试集，验证集,并处理每个分割
        :param matching_pairs:
        :return:
        """
        if not matching_pairs:
            logger.error(f"没有数据可供划分，请检查原始数据集是否正确")
            return

        label_files = [pair[1] for pair in matching_pairs]
        image_files = [pair[0] for pair in matching_pairs]

        if len(matching_pairs) < 3:
            logger.warning(f"数据集样本数量太少：{len(matching_pairs)}，将无法进行有效分割,将所有数据划给训练集")
            self._process_single_split(label_files, image_files, "train")
            return

        # 第一次分割，训练集 vs 临时集 （验证集 + 测试集）
        train_labels, temp_labels, train_images, temp_images = train_test_split(label_files, image_files,
                                                test_size=self.test_rate,
                                                random_state=42,shuffle=True)
        val_labels, test_labels, val_images, test_images = [], [] , [] , []


        # 第二次分割，临时集 （验证集 + 测试集） 内部进行分割
        if temp_labels:
            remaining_rate = self.valid_rate +  self.test_rate
            if remaining_rate == 0 or len(temp_labels) < 2:
                val_labels, val_images = temp_labels, test_images
                logger.warning(f"临时数据集样本数量太少：{len(temp_labels)}，或剩余比例为0，"
                            f"将无法进行有效分割,将所有数据划给验证集")
            else:
                val_ratio_in_temp = self.valid_rate / remaining_rate
                if abs(val_ratio_in_temp) < 1e-6:
                    test_labels, test_images = temp_labels, temp_images
                    logger.info("验证集比例为0，所有剩余数据划给测试集")
                elif abs(val_ratio_in_temp - 1) < 1e-6:
                    val_labels, val_images = temp_labels, temp_images
                    logger.info("测试集比例为0，所有剩余数据划给验证集")
                else:
                    val_labels, test_labels, val_images, test_images = train_test_split(
                        temp_labels, temp_images,
                                test_size=val_ratio_in_temp,
                                random_state=42,shuffle=True)
        logger.info("数据集划分完成")
        logger.info(f"训练集样本数量：{len(train_labels)}")
        logger.info(f"验证集样本数量：{len(val_labels)}")
        logger.info(f"测试集样本数量：{len(test_labels)}")

        self._process_single_split(train_labels, train_images, "train")
        self._process_single_split(val_labels, val_images, "val")
        self._process_single_split(test_labels, test_images, "test")

    def _process_single_split(self, label_files, image_files, split_name):
        """
        处理单个数据集的划分,复制图像和YOLO TXT格式标签到指定的目录
        :param label_files:
        :param image_files:
        :param split_name:
        :return:
        """
        logger.info(f"开始处理：{split_name} 数据集,该数据集共{len(label_files)}个样本")
        target_img_dir = self.output_dirs[split_name]["images"]
        target_label_dir = self.output_dirs[split_name]["labels"]

        target_img_dir.mkdir(parents=True, exist_ok=True)
        target_label_dir.mkdir(parents=True, exist_ok=True)

        copied_images_count = 0
        failed_images_count = 0

        for image_path in image_files:
            new_img_path = target_img_dir / image_path.name
            try:
                shutil.copy(image_path, new_img_path)
                copied_images_count += 1
                logger.debug(f"成功复制图像文件 '{image_path.name}' "
                            f"到 '{new_img_path.relative_to(self.project_root_path)}'")
            except Exception as e:
                failed_images_count += 1
                logger.error(f"复制图像文件 '{image_path.name}' 到 "
                            f"'{new_img_path.relative_to(self.project_root_path)}' 失败: {e}")
        logger.info(f"{split_name} 数据集图像复制完成，成功复制 {copied_images_count} 张，"
                    f"失败复制图像 {failed_images_count} 张")

        copied_labels_count = 0
        failed_labels_count = 0

        for label_path in label_files:
            new_label_path = target_label_dir / label_path.name
            try:
                shutil.copy(label_path, new_label_path)
                copied_labels_count += 1
                logger.debug(f"成功复制标签文件 '{label_path.name}' "
                            f"到 '{new_label_path.relative_to(self.project_root_path)}'")
            except Exception as e:
                failed_labels_count += 1
                logger.debug(f"复制标签文件 '{label_path.name}' 到 "
                            f"'{new_label_path.relative_to(self.project_root_path)}' 失败: {e}")
        logger.info(f"{split_name} 数据集标签复制完成，成功复制标签 {copied_labels_count} 个，"
                    f"失败复制标签 {failed_labels_count} 个")

    def _generate_data_yaml(self):
        """
        生成yaml配置
        :return:
        """
        abs_data_path = self.output_data_path.absolute()
        train_images_abs_path = (self.output_dirs["train"]["images"]).resolve()
        val_images_abs_path = (self.output_dirs["val"]["images"]).resolve()
        test_images_abs_path = (self.output_dirs["test"]["images"]).resolve()

        data_yaml_content = {
            "path": str(abs_data_path),
            "train": str(train_images_abs_path),
            "val": str(val_images_abs_path),
            "test": str(test_images_abs_path),
            "nc": len(self.classes),
            "names": self.classes
        }
        yaml_path = self.config_path / "data.yaml"
        try:
            with open(yaml_path, "w", encoding="utf-8") as f:
                yaml.dump(data_yaml_content, f, default_flow_style=None, sort_keys=False, allow_unicode=True)
                logger.info(f"成功生成 data.yaml 文件：{yaml_path.relative_to(self.project_root_path)}")
                logger.info(f"data.yaml 文件内容："
                f"\n{yaml.dump(data_yaml_content,default_flow_style=None, sort_keys=False, allow_unicode=True)}")
        except Exception as e:
            logger.error(f"生成 data.yaml 文件失败: {e}")
            raise
    @time_it(iterations=1, name="数据集准备与划分", logger_instance=logger)
    def process_dataset(self,source_data_root_dir=ORIGINAL_ANNOTATIONS_DIR,):
        """
        执行整个数据集划分流程
        :param source_data_root_dir:
        :return:
        """
        logger.info("开始进行数据集准备与划分工作".center(60, "="))

        try:
            logger.info(f"处理原始标注数据：{self.annotation_format.upper()}格式")
            if self.annotation_format != "yolo":
                if self.yolo_staged_labels_dir.exists():
                    shutil.rmtree(self.yolo_staged_labels_dir)
                    logger.info(f"已经清理 YOLO 标签暂存目录: "
                                f"{self.yolo_staged_labels_dir.relative_to(self.project_root_path)}")
                    self.yolo_staged_labels_dir.mkdir(parents=True, exist_ok=True)
            if self.annotation_format == "yolo":
                if not self.classes:
                    logger.critical(f"当 annotation_format 为 yolo 是，请务必提供 classes 参数，数据集处理终止")
                    return

                self.yolo_staged_labels_dir = ORIGINAL_ANNOTATIONS_DIR
                logger.info(f"检测到原生的YOLO格式标注文件，YOLO暂存目录直接指向原始标注文件目录"
                        f"'{self.yolo_staged_labels_dir.relative_to(self.project_root_path)}'跳过复制步骤")

                if not any(self.yolo_staged_labels_dir.glob("*.txt")):
                    logger.critical(f"未检测到YOLO格式标注文件，请检查原始标注文件目录："
                                f"'{self.yolo_staged_labels_dir.relative_to(self.project_root_path)}'")
                    return

            elif self.annotation_format in ["coco", "pascal_voc"]:
                if not RAW_IMAGES_DIR.exists() or not any(RAW_IMAGES_DIR.iterdir()):
                    logger.critical(f"未检测到原始图像文件，请检查原始图像存放目录："
                                f"'{RAW_IMAGES_DIR.relative_to(self.project_root_path)}'")
                    return
                if not ORIGINAL_ANNOTATIONS_DIR.exists() or not any(ORIGINAL_ANNOTATIONS_DIR.iterdir()):
                    logger.critical(f"未检测到原始标注文件，请检查原始标注文件存放目录："
                                f"'{ORIGINAL_ANNOTATIONS_DIR.relative_to(self.project_root_path)}'")
                    return
                conversion_input_dir = source_data_root_dir
                self.classes = convert_annotations_to_yolo(
                    input_dir=conversion_input_dir,
                    annotation_format=self.annotation_format,
                    final_classes_order=self.classes if self.annotation_format == "pascal_voc" else None,
                    coco_task=self.coco_task,
                    coco_cls91to80=self.coco_cls91to80
                )
                # 检查最终地转换结果
                if not self.classes:
                    logger.critical(f"{self.annotation_format.upper()}转换失败或未提取到有效的类别信息，数据集处理终止")
                    return
                logger.info(f"{self.annotation_format.upper()}转换成功")
            else:
                logger.critical(f"不支持的标注格式：{self.annotation_format}，数据集处理终止")
                return

            # 调用检查脚本
            self._check_staged_data_existence()

            # 查找匹配的文件
            matching_pairs = self._find_matching_files()
            if not matching_pairs:
                logger.critical(f"未找到匹配的文件，数据集处理终止")
                return

            self._split_and_process_data(matching_pairs)

            # 生成 data.yaml 文件
            self._generate_data_yaml()
        except Exception as e:
            logger.error(f"数据集准备与划分过程发生严重错误: {e}",exc_info=True)
        finally:
            logger.info("数据集准备与划分工作完成".center(60, "="))

def _clean_and_initialize_dirs(processor_instance: YOLODatasetProcessor):
    logger.info("开始清理旧的数据集内容和配置文件".center(60, "="))

    for split_name,split_info in processor_instance.output_dirs.items():
        for dir_type, dir_path in split_info.items():
            if dir_path.exists():
                shutil.rmtree(dir_path, ignore_errors=True)
                logger.info(f"删除已经存在的 '{split_name}' {dir_type}目录：{dir_path.relative_to(YOLOSERVER_ROOT)}")
                dir_path.mkdir(parents=True, exist_ok=True)
                logger.info(f"重新创建 '{split_name}' {dir_type}目录：{dir_path.relative_to(YOLOSERVER_ROOT)}")
    data_yaml_file = CONFIGS_DIR / "data.yaml"
    if data_yaml_file.exists():
        data_yaml_file.unlink()
        logger.info(f"删除已经存在的 data.yaml 文件：{data_yaml_file.relative_to(YOLOSERVER_ROOT)}")
    logger.info("旧数据集内容清理完成，新的目录结构创建完成".center(60, "="))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="YOLO 数据集处理工具",
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("--format", type=str,
                        default="coco",
                        choices=["coco", "pascal_voc", "yolo"],
                        help="支持的数据集标注格式，coco, pascal_voc, yolo")

    parser.add_argument("--train_rate", type=float, default=0.8, help="训练集占比,默认0.8")
    parser.add_argument("--valid_rate", type=float, default=0.1, help="验证集占比,默认0.1")
    parser.add_argument("--classes",type=str,
                        nargs="+", # 允许一个或多个字符串作为列表
                        default=None,
                        help="类别名称列表，以空格分开，例如：--classes class1 class2 class3 \n"
                            "当 --format 为 yolo 时, 必须提供该参数"
                            "当 --format 为 coco 时， 此参数会被忽略"
                            "当 --format 为 pascal_voc 时，可选提供，不指定则使用自动模式"
                        )
    parser.add_argument("--coco_task", type=str,
                        default="segmentation",
                        choices=["detection", "segmentation"],
                        help="COCO任务类型，可选：detection, segmentation")
    parser.add_argument("--coco_cls91to80",default=False,
                        action="store_true", help="将COCO 91类映射 80类")

    args = parser.parse_args()

    processor = YOLODatasetProcessor(train_rate=args.train_rate,
                                    valid_rate=args.valid_rate,
                                    annotation_format=args.format,
                                    final_classes_order=args.classes,
                                    coco_task=args.coco_task,
                                    coco_cls91to80=args.coco_cls91to80
                                    )

    _clean_and_initialize_dirs(processor)

    processor.process_dataset()

    # 打印最终输出结果
    logger.info("所有数据处理流程完成，请检查以下路径文件")
    logger.info(f"训练集图像目录：{processor.output_dirs['train']['images'].relative_to(YOLOSERVER_ROOT)}")
    logger.info(f"训练集标注文件：{processor.output_dirs['train']['labels'].relative_to(YOLOSERVER_ROOT)}")
    logger.info(f"验证集图像目录：{processor.output_dirs['val']['images'].relative_to(YOLOSERVER_ROOT)}")
    logger.info(f"验证集标注文件：{processor.output_dirs['val']['labels'].relative_to(YOLOSERVER_ROOT)}")
    logger.info(f"测试集图像目录：{processor.output_dirs['test']['images'].relative_to(YOLOSERVER_ROOT)}")
    logger.info(f"测试集标注文件：{processor.output_dirs['test']['labels'].relative_to(YOLOSERVER_ROOT)}")
    logger.info(f"数据集配置文件：{processor.config_path.relative_to(YOLOSERVER_ROOT)}")
    logger.info(f"详细的日志文件位于 {LOGS_DIR.relative_to(YOLOSERVER_ROOT)}")
    logger.info(f"接下来请执行数据验证脚本 yolo_validate.py 以验证数据转换是否正确")
